"""Adaptive Importance Sampling (AIS) for Model-Based Diffusion.

Implements the cross-entropy (CE) adaptive importance sampler of

    Golembeski & Mazumdar, "Importance Sampling Model-Based Diffusion for
    Trajectory Optimization", IEEE RA-L 2025 (MBD-AIS).

At each diffusion step, standard MBD draws ``Nsample`` samples from a fixed
isotropic Gaussian ``N(Ybar_i, sigma_i^2 I)``, rolls them out, softmax-weights
them by reward and takes the weighted mean (the Monte-Carlo score estimate,
paper eqs. 6-8).  MBD-AIS instead runs a short cross-entropy loop (paper Alg. 3)
to *adapt* that proposal toward the high-reward region before the final draw
(paper Alg. 2).

This module is deliberately dependency-free (only ``jax``) and rank-agnostic:
it operates on an arbitrary per-sample shape ``S`` through a caller-supplied
``reward_fn``.  The same core therefore serves the single-agent MuJoCo/car
planner in ``mbd`` as well as the multi-car / RRPR / crane planners in
``mrmbd``.

Fidelity note -- the estimator is biased, on purpose
----------------------------------------------------
Only the *proposal* changes.  The caller keeps computing the weighted mean with
the plain reward softmax of paper eq. (6),

    Ybar_0 = sum_j Y_j p_0(Y_j) / sum_j p_0(Y_j),   p_0(Y) ~ exp(-J(Y) / lambda)

with **no importance-sampling likelihood ratio** ``p_i(Y_j) / q(Y_j)``.  Standard
MBD's estimator is consistent because its samples come from the reverse-process
prior ``p_i = N(Ybar_i, sigma_i^2 I)``, so the exponential-reward factor supplies
the likelihood and nothing else is needed.  Once the CE loop moves the proposal
to ``q != p_i``, that identification no longer holds and the score estimate
acquires a bias in the direction the proposal was moved.

The paper does this too, and never mentions it: its Alg. 2 line 5 is "Calculate
Ybar_0 from (6)", identical to standard MBD's Alg. 1 line 4, and the only change
between the two algorithms is the sampling distribution on line 3.  So the
omission here is deliberate fidelity, **not** an oversight -- do not "fix" it by
inserting the ratio, or the numbers stop being comparable to the published ones.
Adding a corrected variant is a fine thing to do; it belongs behind a new,
default-off flag, with the faithful path left intact.
"""

import jax
from jax import numpy as jnp

__all__ = ["ais_adapt", "budget_split"]


def budget_split(n_total: int, n_iter: int):
    """Split a fixed per-step sample budget over the CE iterations (paper eqs 10-12).

    Keeps the total number of reward evaluations equal to ``n_total`` so that
    MBD and MBD-AIS do the *same* amount of work per diffusion step, making a
    reward-vs-``Nsample`` comparison apples-to-apples.

    Args:
        n_total: per-step sample budget (``Nsample``).
        n_iter:  number of cross-entropy adaptation iterations (``Niter``).

    Returns:
        ``(ns_iter, nfinal)`` where ``ns_iter`` samples are used in each of the
        ``n_iter`` CE iterations and ``nfinal`` fresh samples are drawn from the
        adapted proposal for the diffusion averaging step.  When ``n_iter == 0``
        the proposal is untouched and ``nfinal == n_total`` (i.e. standard MBD).

    Note:
        ``ns_iter`` shrinks as ``n_total / (n_iter + 1)``, and the paper flags the
        consequence: below ``n_min`` elites the CE fit degenerates and MBD-AIS is
        unusable.  :func:`ais_adapt` raises rather than silently collapsing.
    """
    if n_iter <= 0:
        return 0, n_total
    ns_iter = n_total // (n_iter + 1)
    nfinal = n_total - ns_iter * n_iter
    return ns_iter, nfinal


def ais_adapt(
    rng,
    mean,
    std0,
    reward_fn,
    ns_iter: int,
    n_iter: int,
    elite_frac: float = 0.1,
    min_elite: int = 2,
    clip=None,
    tau=None,
):
    """Cross-entropy adaptive importance sampling for one MBD reverse step.

    Runs ``n_iter`` CE iterations starting from the standard MBD proposal
    ``N(mean, std0^2 I)`` and returns the adapted diagonal Gaussian.  The caller
    is expected to draw the final ``nfinal`` samples from the returned
    ``(mean, std)`` and continue through the unchanged MBD path (rollout ->
    softmax -> weighted mean -> score -> DDPM update).

    Args:
        rng: JAX PRNGKey.
        mean: initial proposal mean, shape ``S`` == one sample
            (e.g. ``(H, Nu)`` or ``(H, n, Nu)``).
        std0: initial per-element std (scalar ``sigma_i``); broadcast to ``S``.
        reward_fn: callable ``(Y[K, *S]) -> rews[K]``, higher is better.  Must
            include the rollout, horizon-mean and any control clipping.
        ns_iter: samples drawn in each CE iteration (see :func:`budget_split`).
        n_iter: number of CE iterations (``Niter``).  ``0`` returns the initial
            proposal unchanged (=> standard MBD).
        elite_frac: fraction of ``ns_iter`` kept as elites in top-k mode.  Only
            consulted when ``tau is None``.
        min_elite: floor on the number of elites (paper ``n_min``, default 2).
        clip: optional ``(lo, hi)`` bounds applied to drawn samples.
        tau: reward threshold for elite selection.  ``None`` (the default here)
            selects a fixed top-k fraction, CEM-style.  Passing a value selects
            every sample scoring above it, floored at ``min_elite``, which is the
            paper's Alg. 3 rule and the paper's *default*; it is not the default
            here because a useful ``tau`` is problem-specific ("the value of the
            reward expected for the relevant environment"), and a badly chosen one
            silently degrades to keeping exactly ``min_elite`` samples every
            iteration.  Set it to reproduce the paper; leave it unset otherwise.

    Returns:
        ``(mean_adapted[*S], std_adapted[*S], rng)``.

    Raises:
        ValueError: if ``ns_iter`` is too small to fit a non-degenerate
            distribution to the elites.
    """
    S = mean.shape

    # No adaptation: return the proposal untouched. Passing `std0` through
    # unchanged (rather than a broadcast copy) keeps the caller's sampling graph
    # bit-identical to standard MBD, so `ais_niter=0` reproduces MBD exactly.
    if n_iter <= 0 or ns_iter <= 0:
        return mean, std0, rng

    # A variance fitted to fewer than two elites is identically zero, and `std`
    # would collapse to the `sqrt(eps)` floor below -- the proposal dies and every
    # subsequent draw is the same point. The paper flags exactly this regime (its
    # `n_s/iter` falling under `n_min` for large `n_D`), so fail loudly instead of
    # returning a degenerate distribution that still looks like a valid one.
    n_elite_min = max(2, int(min_elite))
    if ns_iter < n_elite_min:
        raise ValueError(
            f"ns_iter={ns_iter} is below min_elite={n_elite_min}: the cross-entropy "
            f"fit degenerates and the proposal collapses to a point. Raise the "
            f"per-step budget to at least {n_elite_min * (int(n_iter) + 1)} samples, "
            f"or lower n_iter."
        )

    # Diagonal proposal std, broadcast from the (typically scalar) sigma_i.
    std = jnp.broadcast_to(jnp.asarray(std0, dtype=mean.dtype), S)

    # Static elite count for top-k mode (jit-safe: fixed array shapes).
    n_elite = min(ns_iter, max(n_elite_min, int(round(elite_frac * ns_iter))))

    for _ in range(int(n_iter)):
        rng, rng_draw = jax.random.split(rng)
        eps = jax.random.normal(rng_draw, (ns_iter,) + S, dtype=mean.dtype)
        Y = mean + std * eps
        if clip is not None:
            Y = jnp.clip(Y, clip[0], clip[1])
        rews = reward_fn(Y)  # (ns_iter,)

        if tau is None:
            # CEM-style: keep the top-k highest-reward elites.
            idx = jnp.argsort(rews)[::-1][:n_elite]
            elite = Y[idx]
            mean = elite.mean(axis=0)
            var = elite.var(axis=0)
        else:
            # Paper Alg. 3: elites are samples above threshold tau, but always
            # keep at least the top `min_elite` (static shape, jit-safe).
            order = jnp.argsort(rews)[::-1]  # best first
            Ys = Y[order]
            rs = rews[order]
            rank = jnp.arange(ns_iter)
            keep = (rs > tau) | (rank < int(min_elite))
            w = keep.astype(mean.dtype)
            w = w / jnp.maximum(w.sum(), 1.0)
            mean = jnp.einsum("k,k...->...", w, Ys)
            var = jnp.einsum("k,k...->...", w, (Ys - mean) ** 2)

        # Plain MLE fit, no shrinkage and no convex blend with the previous
        # iterate: the paper contrasts its MLE update with MPOPI-CE's
        # Schaefer-Strimmer shrinkage on purpose, so there is no smoothing
        # factor to tune here. The contraction is steep -- measured on the
        # multi-car pillar scenario at Nsample=1024, n_iter=2, std goes
        # 1.0 -> ~0.2 over the two iterations. That is intended behaviour, but
        # it is also why `ns_iter` has to stay comfortably above `min_elite`.
        std = jnp.sqrt(var + 1e-8)

    return mean, std, rng
