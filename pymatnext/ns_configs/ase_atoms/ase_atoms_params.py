"""dict with template of parameters for [configs] and [configs.walk] sections
"""

param_defaults_ase_atoms = {
    "full_composition": "",
    "seed_config": "_IGNORE_",
    "composition": ["_REQ_", "AbCdE2"],
    "n_atoms": ["_REQ_", 1],
    "dims": 3,
    "pbc": [True, True, True],
    "initial_rand_vol_per_atom": ["_REQ_", 1.0],
    "initial_rand_min_dist": ["_REQ_",{ "_IGNORE_": 1.0 }],
    "initial_rand_n_tries": 10,
    "limit": [0., 1.],
    "calculator": {
        "type": ["_REQ_", "calc_type"],
        "args": { "_IGNORE_": True }
    },
}

