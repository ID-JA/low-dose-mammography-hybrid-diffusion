from types import SimpleNamespace

"""
    -- VQ-VAE-2 Hyperparameters --
"""
_common = {
    'checkpoint_frequency':         4,
    'image_frequency':              1,
    'test_size':                    0.1,
    'nb_workers':                   1,
}

_cbis_ddsm = {
    'display_name':             'CBIS-DDSM',
    'image_shape':              (1, 1024, 768),

    'in_channels':              1,
    'hidden_channels':          128,
    'res_channels':             32,
    'nb_res_layers':            2,
    'embed_dim':                64,
    'nb_entries':               512,
    'nb_levels':                5,
    'scaling_rates':            [4, 2, 2, 2, 2],

    'learning_rate':            1e-4,
    'beta':                     0.25,
    'batch_size':               4,
    'mini_batch_size':          4,
    'max_epochs':               100,
}

HPS_VQVAE = {
    'cbis-ddsm':            SimpleNamespace(**(_common | _cbis_ddsm)),
}

_diffusion_common = {
    # --- noise schedule ---
    'timesteps':          1000,
    'beta_schedule':      'linear',
    'linear_start':       0.0015,
    'linear_end':         0.0195,
    'scale_factor':       1.0,

    # --- UNet architecture ---
    'model_channels':     192,          # base channel width
    'channel_mult':       [1, 2, 4],    # → 192, 384, 768
    'num_res_blocks':     2,
    'attention_levels':   [2],          # self-attn at deepest level (32×24)
    'dropout':            0.0,
    'num_head_channels':  64,           # → heads = ch // 64
    'use_checkpoint':     False,        # gradient checkpointing (saves VRAM)

    # --- training ---
    'learning_rate':      1e-4,
    'batch_size':         2,
    'mini_batch_size':    2,
    'max_epochs':         200,
    'grad_clip_norm':     1.0,          # max gradient norm (official LDM uses 1.0)

    # --- EMA  (matches official LDM defaults) ---
    'use_ema':            True,
    'ema_decay':          0.9999,

    # --- loss weighting  (matches ddpm.py defaults) ---
    'learn_logvar':       False,
    'logvar_init':        0.0,
    'l_simple_weight':    1.0,
    'original_elbo_weight': 0.0,        # VLB term weight (0 = disabled)

    # --- DDIM sampling ---
    'ddim_steps':         50,
    'ddim_eta':           0.0,
    'unconditional_guidance_scale': 1.0,  # >1.0 enables classifier-free guidance
}

HPS_DIFFUSION = {
    'cbis-ddsm': SimpleNamespace(**(_common | _cbis_ddsm | _diffusion_common)),
}
