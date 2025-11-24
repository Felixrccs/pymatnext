"""dict with template of parameters for [ns] section
"""

param_defaults = {
    "n_walkers": ["_REQ_", 1],
    "walk_length": ["_REQ_", 1],
    "parallel_walks": ["_REQ_", 8],
    "configs_module": ["_REQ_", "ns_config_mod"],
    "exit_conditions": { "_IGNORE_": True },
    "initial_config_file": "_NONE_"
}
