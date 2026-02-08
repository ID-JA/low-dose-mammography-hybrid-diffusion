from types import SimpleNamespace

"""
    -- VQ-VAE-2 Hyperparameters --
"""
_common = {
    'checkpoint_frequency':         4,
    'image_frequency':              1,
    'test_size':                    0.1,
    'nb_workers':                   4,
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
    'nb_levels':                3,
    'scaling_rates':            [8, 2, 2],

    'learning_rate':            1e-4,
    'beta':                     0.25,
    'batch_size':               4,
    'mini_batch_size':          4,
    'max_epochs':               100,
}

HPS_VQVAE = {
    'cbis-ddsm':            SimpleNamespace(**(_common | _cbis_ddsm)),
}
