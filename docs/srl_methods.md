# SRL Methods

This release keeps the SRL objectives reported in the DVP paper.

| CLI name | Paper name | Training signal |
| --- | --- | --- |
| `ppo` | PPO | No auxiliary representation objective. |
| `ppo_dvp` | DVP | Aligns deployable actor observations with training-only privileged observations in latent space. |
| `ppo_spr` | SPR | Predicts future latent states from current deployable observations and actions. |
| `ppo_vae` | VAE | Decodes privileged-state targets from deployable actor-side latents with a variational bottleneck. |
| `ppo_simsiam` | SimSiam | Aligns two perturbed deployable observation views with stop-gradient cosine matching. |

The shared controls are:

```text
--srl_algo_name
--srl_loss_coef
--srl_time_prop
--srl_data_prop
--srl_interval
```

Method-specific controls:

```text
--dvp_loss_coef
--dvp_hidden_dim
--dvp_view_mode
--dvp_privileged_position

--spr_loss_coef
--spr_k
--spr_skip
--spr_tau
--spr_hidden_dim
--spr_projector_dim
--spr_aug

--vae_loss_coef
--vae_latent_dim
--vae_hidden_dim
--vae_kl_weight

--simsiam_loss_coef
--simsiam_hidden_dim
--srl_aug_q
--srl_aug_k
```

`ppo_dvp` is the DVP method used in the paper. It uses the same rollout action
path as PPO and uses privileged information only for the auxiliary latent
alignment loss during training.
