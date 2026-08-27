import functools
import os
import jax
from jax import numpy as jnp
from jax import config
from dataclasses import dataclass
from typing import Literal
import tyro
from tqdm import tqdm
from matplotlib import pyplot as plt

import mbd

# NOTE: enable this if you want higher precision
# config.update("jax_enable_x64", True)


## load config
@dataclass
class Args:
    # exp
    seed: int = 0
    disable_recommended_params: bool = False
    not_render: bool = False
    # env
    env: str = (
        "ant"  # "humanoidstandup", "ant", "halfcheetah", "hopper", "walker2d", "car2d"
    )
    # diffusion
    Nsample: int = 2048  # number of samples
    Hsample: int = 50  # horizon
    Ndiffuse: int = 100  # number of diffusion steps
    temp_sample: float = 0.1  # temperature for sampling
    beta0: float = 1e-4  # initial beta
    betaT: float = 1e-2  # final beta
    enable_demo: bool = False
    # Adaptive Importance Sampling (MBD-AIS, Golembeski & Mazumdar RA-L 2025)
    ais: bool = False           # replace the fixed Gaussian sampler with a CE adaptive importance sampler
    ais_niter: int = 2          # cross-entropy adaptation iterations per diffusion step (Niter); 0 == standard MBD
    ais_elite_frac: float = 0.1 # fraction of samples kept as elites each CE iteration (top-k mode)
    ais_min_elite: int = 2      # floor on the number of elites (paper n_min)
    ais_tau: float | None = None  # optional reward threshold for paper-faithful elite selection (None => top-k)
    # PID Langevin Dynamics
    pid: bool = False       # enable PID-controlled score update
    pid_schedule: Literal["none", "snr", "ess"] = "none"  # gain scheduling mode (implies --pid)
    kp: float = 1.0         # proportional gain
    ki: float = 0.1         # integral gain
    kd: float = 0.05        # derivative gain
    gamma: float = 0.95     # integral gain decay per step
    # Underdamped Langevin Dynamics
    underdamped: bool = False      # enable momentum-augmented reverse diffusion
    friction: float = 0.5          # damping coefficient γ (0 = no friction, ∞ = overdamped)
    mass: float = 1.0              # effective mass (scales velocity inertia)
    velocity_clip: float = 2.0     # max velocity magnitude (prevents runaway)
    # Adam-Langevin (SamAdams) adaptive stepsize
    adam_langevin: bool = False
    al_alpha: float = 1.0       # attack rate (EMA decay)
    al_omega: float = 400.0     # monitor scale (≈ Hsample*Nu; normalizes schedule-corrected score norm)
    al_m: float = 0.5           # min stepsize factor
    al_M: float = 2.0           # max stepsize factor
    al_r: float = 0.25          # power in Sundman kernel
    al_s: float = 2.0           # power in monitor (2 = squared norm, Adam-like)
    al_kernel: int = 2          # 1 = ψ^(1), 2 = ψ^(2)
    # Smoothness penalties
    smooth_fd: bool = False            # enable finite-difference smoothness penalty
    smooth_fd_weight: float = 0.1      # weight for FD penalty
    smooth_bw: bool = False            # enable FFT high-frequency penalty
    smooth_bw_weight: float = 0.1      # weight for FFT penalty
    smooth_bw_cutoff: float = 0.3      # fraction of freq bins considered "high" (0=all, 1=none)


def run_diffusion(args: Args):

    if args.pid_schedule != "none":
        args.pid = True  # pid_schedule implies pid
    if args.pid and args.underdamped:
        raise ValueError("Cannot use both --pid and --underdamped. Choose one.")
    if args.adam_langevin and (args.pid or args.underdamped):
        raise ValueError("Cannot use --adam_langevin with --pid or --underdamped.")

    rng = jax.random.PRNGKey(seed=args.seed)

    ## setup env

    # recommended temperature for envs
    temp_recommend = {
        "ant": 0.1,
        "halfcheetah": 0.4,
        "hopper": 0.1,
        "humanoidstandup": 0.1,
        "humanoidrun": 0.1,
        "walker2d": 0.1,
        "pushT": 0.2,
    }
    Ndiffuse_recommend = {
        "pushT": 200,
        "humanoidrun": 300,
    }
    Nsample_recommend = {
        "humanoidrun": 8192,
    }
    Hsample_recommend = {
        "pushT": 40,
    }
    if not args.disable_recommended_params:
        args.temp_sample = temp_recommend.get(args.env, args.temp_sample)
        args.Ndiffuse = Ndiffuse_recommend.get(args.env, args.Ndiffuse)
        args.Nsample = Nsample_recommend.get(args.env, args.Nsample)
        args.Hsample = Hsample_recommend.get(args.env, args.Hsample)
        print(f"override temp_sample to {args.temp_sample}")
    env = mbd.envs.get_env(args.env)
    Nx = env.observation_size
    Nu = env.action_size
    # env functions
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    # eval_us = jax.jit(functools.partial(mbd.utils.eval_us, step_env_jit))
    rollout_us = jax.jit(functools.partial(mbd.utils.rollout_us, step_env_jit))

    rng, rng_reset = jax.random.split(rng)  # NOTE: rng_reset should never be changed.
    state_init = reset_env_jit(rng_reset)

    ## run diffusion

    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)
    Sigmas_cond = (
        (1 - alphas) * (1 - jnp.sqrt(jnp.roll(alphas_bar, 1))) / (1 - alphas_bar)
    )
    sigmas_cond = jnp.sqrt(Sigmas_cond)
    sigmas_cond = sigmas_cond.at[0].set(0.0)
    print(f"init sigma = {sigmas[-1]:.2e}")

    YN = jnp.zeros([args.Hsample, Nu])

    @jax.jit
    def reverse_once(carry, unused):
        if args.pid:
            i, rng, Ybar_i, I_accum, s_prev = carry
        elif args.underdamped:
            i, rng, Ybar_i, velocity = carry
        elif args.adam_langevin:
            i, rng, Ybar_i, zeta = carry
        else:
            i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        def reward_fn(Y):
            r, _ = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y)
            return r.mean(axis=-1)

        # sample from q_i (optionally via CE adaptive importance sampling, MBD-AIS)
        rng, Y0s_rng = jax.random.split(rng)
        if args.ais:
            ns_iter, nfinal = mbd.ais.budget_split(args.Nsample, args.ais_niter)
            mean_u, std_u, Y0s_rng = mbd.ais.ais_adapt(
                Y0s_rng, Ybar_i, sigmas[i], reward_fn,
                ns_iter, args.ais_niter,
                elite_frac=args.ais_elite_frac, min_elite=args.ais_min_elite,
                clip=(-1.0, 1.0), tau=args.ais_tau,
            )
            ndraw = nfinal
        else:
            mean_u, std_u, ndraw = Ybar_i, sigmas[i], args.Nsample
        eps_u = jax.random.normal(Y0s_rng, (ndraw, args.Hsample, Nu))
        Y0s = eps_u * std_u + mean_u
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # esitimate mu_0tm1
        rewss, qs = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
        rews = rewss.mean(axis=-1)

        # Smoothness penalties
        if args.smooth_fd:
            diffs = Y0s[:, 1:, :] - Y0s[:, :-1, :]
            fd_penalty = (diffs ** 2).sum(axis=(-1, -2))
            rews = rews - args.smooth_fd_weight * fd_penalty

        if args.smooth_bw:
            freqs = jnp.fft.rfft(Y0s, axis=1)
            n_freq = freqs.shape[1]
            cutoff_idx = max(1, int(n_freq * args.smooth_bw_cutoff))
            high_freq_energy = (jnp.abs(freqs[:, cutoff_idx:, :]) ** 2).sum(axis=(-1, -2))
            rews = rews - args.smooth_bw_weight * high_freq_energy

        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        rew_mean = rews.mean()
        logp0 = (rews - rew_mean) / rew_std / args.temp_sample

        # evalulate demo
        if args.enable_demo:
            xref_logpds = jax.vmap(env.eval_xref_logpd)(qs)
            xref_logpds = xref_logpds - xref_logpds.max()
            logpdemo = (
                (xref_logpds + env.rew_xref - rew_mean) / rew_std / args.temp_sample
            )
            demo_mask = logpdemo > logp0
            logp0 = jnp.where(demo_mask, logpdemo, logp0)
            logp0 = (logp0 - logp0.mean()) / logp0.std() / args.temp_sample

        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)  # NOTE: update only with reward

        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)

        if args.pid:
            P = score
            step = args.Ndiffuse - 1 - i          # counts 0, 1, 2, ...
            I_new = (I_accum * step + score) / (step + 1)  # running average
            D = score - s_prev
            if args.pid_schedule == "snr":
                # SNR-based gain scheduling: ramp gains with signal clarity
                snr_i = alphas_bar[i] / (1.0 - alphas_bar[i] + 1e-8)
                snr_weight = snr_i / (1.0 + snr_i)  # ∈ [0,1]
                kp_t = args.kp
                ki_t = args.ki * snr_weight
                kd_t = args.kd * snr_weight
            elif args.pid_schedule == "ess":
                # ESS-based gain scheduling: adapt to optimization state
                ess = 1.0 / jnp.sum(weights ** 2)
                ess_weight = ess / args.Nsample
                kp_t = args.kp
                ki_t = args.ki * ess_weight
                kd_t = args.kd * ess_weight
            else:
                kp_t = args.kp
                ki_t = args.ki * (args.gamma ** step)
                kd_t = args.kd
            u = kp_t * P + ki_t * I_new + kd_t * D
            Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * u)
        elif args.underdamped:
            score_force = (1.0 - alphas_bar[i]) * score
            velocity_new = (1.0 - args.friction) * velocity + (1.0 / args.mass) * score_force
            velocity_new = jnp.clip(velocity_new, -args.velocity_clip, args.velocity_clip)
            Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + velocity_new)
        elif args.adam_langevin:
            score_norm = jnp.sqrt(jnp.sum(score ** 2) + 1e-12)
            # Factor out noise-schedule magnitude so the monitor tracks
            # reward-landscape difficulty, not diffusion step position.
            schedule_norm = score_norm * (1.0 - alphas_bar[i])
            g_val = (schedule_norm ** args.al_s) / args.al_omega
            rho_half = jnp.exp(-args.al_alpha * 0.5)  # Δτ = 1
            zeta_half = rho_half * zeta + (1.0 - rho_half) / args.al_alpha * g_val
            if args.al_kernel == 1:
                psi_val = args.al_m * (zeta_half ** args.al_r + args.al_M) / (zeta_half ** args.al_r + args.al_m)
            else:
                psi_val = args.al_m * (zeta_half ** args.al_r + args.al_M / args.al_m) / (zeta_half ** args.al_r + 1.0)
            Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + psi_val * (1.0 - alphas_bar[i]) * score)
            zeta_new = rho_half * zeta_half + (1.0 - rho_half) / args.al_alpha * g_val
        else:
            Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)

        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])

        if args.pid:
            return (i - 1, rng, Ybar_im1, I_new, score), rews.mean()
        elif args.underdamped:
            return (i - 1, rng, Ybar_im1, velocity_new), rews.mean()
        elif args.adam_langevin:
            return (i - 1, rng, Ybar_im1, zeta_new), rews.mean()
        else:
            return (i - 1, rng, Ybar_im1), rews.mean()

    # run reverse
    def reverse(YN, rng):
        Yi = YN
        Ybars = []
        rew_history = []
        if args.pid:
            I_accum = jnp.zeros_like(YN)
            s_prev = jnp.zeros_like(YN)
        elif args.underdamped:
            velocity = jnp.zeros_like(YN)
        elif args.adam_langevin:
            zeta = 0.0
        with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                if args.pid:
                    carry_once = (i, rng, Yi, I_accum, s_prev)
                    (i, rng, Yi, I_accum, s_prev), rew = reverse_once(carry_once, None)
                elif args.underdamped:
                    carry_once = (i, rng, Yi, velocity)
                    (i, rng, Yi, velocity), rew = reverse_once(carry_once, None)
                elif args.adam_langevin:
                    carry_once = (i, rng, Yi, zeta)
                    (i, rng, Yi, zeta), rew = reverse_once(carry_once, None)
                else:
                    carry_once = (i, rng, Yi)
                    (i, rng, Yi), rew = reverse_once(carry_once, None)
                Ybars.append(Yi)
                rew_history.append(float(rew))
                pbar.set_postfix({"rew": f"{rew:.2e}"})
        return jnp.array(Ybars), rew_history

    rng_exp, rng = jax.random.split(rng)
    Yi, rew_history = reverse(YN, rng_exp)
    if not args.not_render:
        path = f"{mbd.__path__[0]}/../results/{args.env}"
        if not os.path.exists(path):
            os.makedirs(path)
        jnp.save(f"{path}/mu_0ts.npy", Yi)
        if args.env == "car2d":
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            # rollout
            xs = jnp.array([state_init.pipeline_state])
            state = state_init
            for t in range(Yi.shape[1]):
                state = step_env_jit(state, Yi[-1, t])
                xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
            env.render(ax, xs)
            if args.enable_demo:
                ax.plot(env.xref[:, 0], env.xref[:, 1], "g--", label="RRT path")
            ax.legend()
            plt.savefig(f"{path}/rollout.png")
        else:
            render_us = functools.partial(
                mbd.utils.render_us,
                step_env_jit,
                env.sys.tree_replace({"opt.timestep": env.dt}),
            )
            webpage = render_us(state_init, Yi[-1])
            with open(f"{path}/rollout.html", "w") as f:
                f.write(webpage)
    rewss_final, _ = rollout_us(state_init, Yi[-1])
    rew_final = rewss_final.mean()

    return rew_final, rew_history


if __name__ == "__main__":
    rew_final, _ = run_diffusion(args=tyro.cli(Args))
    print(f"final reward = {rew_final:.2e}")
