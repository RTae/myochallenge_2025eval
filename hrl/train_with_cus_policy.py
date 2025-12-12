from sb3_contrib import RecurrentPPO
from lattice.ppo.policies import LatticeRecurrentActorCriticPolicy
from lattice.sac.policies import LatticeSACPolicy

    # --- Lattice SAC ---
    # model = SAC(policy=LatticeSACPolicy,
    #     env=vec_env,
    #     device='auto',
    #     learning_rate=3e-4,
    #     buffer_size=300_000,
    #     learning_starts=10000,
    #     batch_size=256,
    #     tau=0.02,
    #     gamma=0.98,
    #     train_freq=(8, "step"),
    #     gradient_steps=8,
    #     action_noise=None,
    #     replay_buffer_class=None,
    #     ent_coef="auto",
    #     target_update_interval=1,
    #     target_entropy="auto",
    #     use_sde=False,
    #     sde_sample_freq=1,
    #     policy_kwargs=dict(
    #         **policy_kwargs,
    #         use_lattice=True,
    #         use_expln=True,
    #         log_std_init=0.0,
    #         # activation_fn=nn.GELU,
    #         std_clip=(1e-3, 1),
    #         expln_eps=1e-6,
    #         clip_mean=2.0,
    #         std_reg=0.0
    #     ),)
    
    # --- Lattice PPO ---
    # model = RecurrentPPO(policy=LatticeRecurrentActorCriticPolicy, 
    #     env=vec_env,
    #     device='auto',
    #     batch_size=32,
    #     n_steps=128,
    #     learning_rate=2.55673e-05,
    #     ent_coef=3.62109e-06,
    #     clip_range=0.3,
    #     gamma=0.99,
    #     gae_lambda=0.9,
    #     max_grad_norm=0.7,
    #     vf_coef=0.835671,
    #     n_epochs=10,
    #     use_sde=False,
    #     sde_sample_freq=1,
    #     policy_kwargs=dict(
    #         **policy_kwargs,
    #         use_lattice=True,
    #         use_expln=True,
    #         ortho_init=False,
    #         log_std_init=0.0,
    #         # activation_fn=nn.ReLU,
    #         std_clip=(1e-3, 10),
    #         expln_eps=1e-6,
    #         full_std=False,
    #         std_reg=0.0,
    #     ),)
    