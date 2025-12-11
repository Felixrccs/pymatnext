import importlib

import re
from pathlib import Path

import json
import copy

import numpy as np

from .ns_utils import rngs as new_rngs

from pymatnext.params import check_fill_defaults
from .ns_params import param_defaults

from .ns_configs.ase_atoms.walks_torch import state_init, append_state_to_atoms, torch_walker


class NS:
    """Nested sampling object

    Parameters
    ----------
    params_ns: dict
        setup parameters
    comm: Communicator
        communicator for parallelism
    MPI: mpi4py.MPI or equivalent namespace
        namespace needed for symbols require to call MPI functions
    random_seed: int or None
        random seed for RNGs
    params_config: dict
        dict for [configs] toml section
    output_filename_prefix: str / Path
        prefix of all output filenames
    different_n_rng_local: bool, default False
        allow restart from snapshot even if number of local rngs stored does not match number this time
    extra_config: bool, default False
        allocate storage for an extra config, e.g. to use as a buffer
    """

    def __init__(
        self,
        params_ns,
        comm,
        MPI,
        random_seed,
        params_configs,
        output_filename_prefix,
        different_n_rng_local=False,
        extra_config=False,
    ):
        check_fill_defaults(params_ns, param_defaults, label="ns")

        self.comm = comm
        self.MPI = MPI
        self.n_configs_global = params_ns["n_walkers"]
        self.global_walk_length = params_ns["walk_length"]
        self.parallel = params_ns["parallel_walks"]

        # get configuration constructor from module that defines exactly one class whose name starts with NSConfig_
        print("###### configs_module ########")
        nsconfig_mod = importlib.import_module(params_ns["configs_module"])
        nsconfig_classes = [
            symb for symb in dir(nsconfig_mod) if symb.startswith("NSConfig_")
        ]
        assert (
            len(nsconfig_classes) == 1
        )  # Internal: check if only one config mode importet
        self.NSConfig = getattr(nsconfig_mod, nsconfig_classes[0])
        self.NSConfig.initialize(params_configs)

        # local walk is shortened by factor of number of parallel processes
        self.local_walk_length = int(self.global_walk_length / comm.size)

        # allocate correct numbers of configs to each node
        if self.n_configs_global % self.comm.size == 0:
            self.n_configs_local = self.n_configs_global // self.comm.size
            self.max_n_configs_local = self.n_configs_global // self.comm.size
            # self.n_configs_global_offset = self.n_configs_local * self.comm.rank
        else:
            raise ValueError(
                f"Number of configurations {self.n_configs_global} must be divisible by number of processes {self.comm.size}"
            )

        # number of quantities depends on the type of config, so these must want for call to init_configs()
        self._allgatherv_counts = None
        self._allgatherv_displt = None
        self._ns_quants_global = None

        # set when find_max() is called
        self.rank_of_max = None
        self.local_ind_of_max = None
        self.max_val = None

        self.local_configs = []
        self.extra_config = False

        old_state_files = NS._old_state_files(output_filename_prefix)
        if len(old_state_files) > 0:
            # found a snapshot
            snapshot_state_file = old_state_files[-1]
            self.snapshot_iter = NS._iter_from_state_file(snapshot_state_file)

            initial_config_file = snapshot_state_file.replace(
                ".state.json", f".configs{self.NSConfig.filename_suffix}"
            )
            with open(snapshot_state_file) as fin:
                snapshot_state = json.load(fin)
        else:
            # no snapshot, generate from scratch
            snapshot_state = {}
            if params_ns["initial_config_file"] != "_NONE_":
                initial_config_file = params_ns["initial_config_file"]
            else:
                initial_config_file = None
            self.snapshot_iter = -1

        self.init_rngs(
            random_seed,
            snapshot_state.get("rngs", None),
            different_nlocal=different_n_rng_local,
        )
        self.init_configs(params_configs, initial_config_file, extra=extra_config)

    def get_calculator(self):
        calc_module = importlib.import_module('examples.ASE_style_calc.MACE_torch')
        return calc_module.calc

    def report_store(self, loop_iter):
        """Store quantities needed for NSConfig-specific report on progress of NS iteration

        Parameters
        ----------
        loop_iter: int
            current NS loop iteration number
        """
        return self.NSConfig.report_store(loop_iter, self.max_val)

    def report(self):
        """Write report on progress of NS iteration

        Returns
        -------
        report_str: text line with report info
        """
        return self.NSConfig.report()

    @staticmethod
    def _iter_from_state_file(filename):
        """extract iteration number from snapshot filename

        Parameters
        ----------
        filename: str
            filename

        Returns
        -------
        iter_i: int iteration number
        """
        return int(re.sub(".state.json", "", re.sub(r".*iter_", "", filename)))

    @staticmethod
    def _old_state_files(output_filename_prefix):
        """get sorted list of old state snapshot files

        Parameters
        ----------
        output_filename_prefix: str
            prefix of output filenames

        Returns
        -------
        old_state_files: list(str) filenames, in order of increasing iteration
        """
        if output_filename_prefix is None:
            return []

        prefix_path = Path(output_filename_prefix)

        old_state_files = [
            str(f)
            for f in prefix_path.parent.glob(prefix_path.name + ".iter_*.state.json")
        ]
        old_state_files = sorted(
            old_state_files, key=lambda filename: NS._iter_from_state_file(filename)
        )

        return old_state_files

    def init_rngs(
        self, random_seed=None, bit_generator_states=None, different_nlocal=False
    ):
        """Initialize rngs from previous state in a dict, or from a seed

        Parameters
        ----------
        random_seed: int, default None
            seed to use (if state is not provided)
        bit_generator_states: dict, default None
            dict with saved rng states ``{"global": dict,  "locals": [dict, ...]}``,
            overrides ``random_seed``
        different_nlocal: bool, default False
            if bit_generator_states is provided, use it as much as possible even if
            it provides a mismatched number of local bit_generators
        """
        # construct rng objects
        self.rng_global, self.rng_local = new_rngs(self.comm.rank, random_seed)

        # override state if provided
        if bit_generator_states is not None:
            self.read_rngs(bit_generator_states, different_nlocal=different_nlocal)

    def init_configs(self, params_configs, configs_file=None, extra=False):
        """initialize all configurations by reading or constructing objects on head node and
        sending them to each compute node

        Sets self.snapshot_iter > 0 if a snapshot was read

        Parameters
        ----------
        params_configs: dict
            dictionary of parameters for configurations created by NSConfig method
        extra: bool, default False
            construct an extra config to use for pilot walks
        """

        if configs_file is None:
            assert self.snapshot_iter < 0

        # define generators for new configs from file or randomly generated
        new_configs_generator = self.NSConfig.new_configs_generator(
            self.n_configs_global, copy.deepcopy(params_configs), self.rng_global, configs_file
        )

        # generate on root, send to each node
        self.local_configs = []
        if self.comm.rank == 0:
            for config_i, new_config in enumerate(new_configs_generator()):
                if config_i == 0:
                    first_config = new_config
                if config_i >= self.n_configs_global:
                    raise RuntimeError(
                        f"Got too many configs (expected {self.n_configs_global}) from new config generator {new_configs_generator}"
                    )

                # Check that all step sizes are the same. Maybe instead we should just copy from first?
                assert (
                    new_config.step_size == first_config.step_size
                ), f"Mismatched step size for config {config_i} {new_config.step_size} != 0 {first_config.step_size}"

                target_rank = config_i // self.max_n_configs_local
                if target_rank == self.comm.rank:
                    self.local_configs.append(new_config)
                else:
                    self.comm.send(
                        new_config,
                        target_rank,
                        tag=15 + config_i % self.max_n_configs_local,
                    )

            if config_i + 1 != self.n_configs_global:
                raise RuntimeError(
                    f"Got not enough configs ({config_i + 1}) from config generator, expected {self.n_configs_global}"
                )
        else:
            for config_i in range(self.n_configs_local):
                self.local_configs.append(self.comm.recv(source=0, tag=15 + config_i))

        models = self.get_calculator()
        self.walker = torch_walker(model=models[1], E_model=models[0], atoms=self.local_configs[0].atoms)
        
        # batch inital calculations to spare atoms
        for atoms_ids in np.array_split(np.arange(self.n_configs_global),self.n_configs_global//10):
            state = state_init([self.local_configs[local_i].atoms for local_i in atoms_ids], models[0])
            append_state_to_atoms(state, [self.local_configs[local_i].atoms for local_i in atoms_ids])
        print('local_info', self.local_configs[20].atoms.info)

        # prepare all configs for NS simulation
        for i, local_config in enumerate(self.local_configs):
            local_config.prepare()

        
        

        # NOTE: this really belongs with the class, not the individual config
        # need to move initialization of the Zs to a classmethod
        n_quantities = self.local_configs[0].n_quantities

        self._allgatherv_counts = [
            self.max_n_configs_local * n_quantities
        ] * self.comm.size
        self._allgatherv_displs = [
            i * (self.max_n_configs_local * n_quantities) for i in range(self.comm.size)
        ]

        self._ns_quants_global = np.finfo(np.float64).min * np.ones(
            (self.comm.size, self.max_n_configs_local, n_quantities)
        )

        

        if extra:
            # construct extra config to use as a buffer, contents don't matter
            self.extra_config = copy.deepcopy(new_config)
            self.extra_config.atoms.positions = np.zeros_like(self.extra_config.atoms.positions)
        
        # re-sync after root process used rng_global to generate configs
        # Todo: sync random generator, wierd bug
        self.rng_global.bit_generator.state = self.comm.bcast(
            self.rng_global.bit_generator.state, root=0
        )

    def find_max(self):
        """Find maximum of nested sampling quantity, as well as its location (parallel process rank and
        local index), and other (system specific) quantities

        Stores in self.rank_of_max, self.local_ind_of_max, self.max_val
        """
        # gather all quantities so all nodes can do the same maximization
        # NOTE: if bandwidth is limiting, rather than latency (which seems unlikely),
        # might be faster to gather only NS quantity, find max location, then bcast all
        # other quantities from that location
        self._ns_quants_global[self.comm.rank][: self.n_configs_local][:] = [
            config.ns_quantities() for config in self.local_configs
        ]
        self.comm.Allgatherv(
            self.MPI.IN_PLACE,
            [
                self._ns_quants_global,
                self._allgatherv_counts,
                self._allgatherv_displs,
                self.MPI.DOUBLE,
            ],
        )

        # find max of NS quantity (index 0 in each config's quantities) with pure numpy ops

        # local index of max value on each proc
        local_ind_of_max_in_each_proc = np.argmax(
            self._ns_quants_global[:, :, 0], axis=1
        )
        # max value on each proc
        val_of_max_in_each_proc = np.max(self._ns_quants_global[:, :, 0], axis=1)

        # rank of proc that has global max
        self.rank_of_max = np.argmax(val_of_max_in_each_proc)
        # index in proc that has global max
        self.local_ind_of_max = local_ind_of_max_in_each_proc[self.rank_of_max]
        # global max value
        self.max_val = val_of_max_in_each_proc[self.rank_of_max]
        # other quantities of max value config
        self.max_quants = self._ns_quants_global[
            self.rank_of_max, self.local_ind_of_max, 1:
        ]

    def global_ind(self, rank, local_ind):
        """global index corresponding to a rank and local index

        Parameters
        ----------
        rank: int
            rank of process
        local_ind: int
            local index in rank

        Returns
        -------
        global_ind: global index
        """
        return rank * self.max_n_configs_local + local_ind

    def local_ind(self, global_ind):
        """rank and local index corresponding to a global index

        Parameters
        ----------
        global_ind: int
            global index

        Returns
        --------
        rank, local_ind: rank and local index
        """

        return (
            global_ind // self.max_n_configs_local,
            global_ind % self.max_n_configs_local,
        )


    def snapshot(self, loop_iter, output_filename_prefix, save_old=2):
        """write a snapshot of the NS system

        Parameters
        ----------
        loop_iter: int
            iteration of NS loop
        output_filename_prefix: str
            initial part of filenames that will be written to
        save_old: int, default 2
            number of old snapshots to save
        """
        if self.comm.rank == 0:
            old_state_files = NS._old_state_files(output_filename_prefix)
            if len(old_state_files) > save_old - 1:
                old_state_files = old_state_files[: -(save_old - 1)]
            else:
                old_state_files = []

        output_filename_prefix_iter = f"{output_filename_prefix}.iter_{loop_iter}"

        config_suffix = self.NSConfig.filename_suffix

        # save snapshots
        if self.comm.rank == 0:
            with open(
                output_filename_prefix_iter + f".configs{config_suffix}", "w"
            ) as fout:
                # write own data
                for local_config in self.local_configs:
                    local_config.write(
                        fout, extra_info={"from_rank": 0}, full_state=True
                    )

                # receive from each
                for remote_config_i in range(
                    self.n_configs_local, self.n_configs_global
                ):
                    source_rank = remote_config_i // self.max_n_configs_local
                    self.extra_config.recv(source_rank, self.comm, self.MPI)
                    self.extra_config.write(
                        fout, extra_info={"from_rank": source_rank}, full_state=True
                    )
        else:
            for local_config in self.local_configs:
                local_config.send(0, self.comm, self.MPI)

        # gather states of all local rngs
        state = {}
        state["rngs"] = self.write_rngs()
        if self.comm.rank == 0:
            with open(output_filename_prefix_iter + ".state.json", "w") as fout:
                json.dump(state, fout)

        # make sure writes did something
        if self.comm.rank == 0:
            if (
                Path(output_filename_prefix_iter + f".configs{config_suffix}")
                .stat()
                .st_size
                <= 0
            ):
                raise RuntimeError(
                    f"Failed to write to snapshot {output_filename_prefix_iter}.configs{config_suffix} file, refusing to continue"
                )
            if Path(output_filename_prefix_iter + ".state.json").stat().st_size <= 0:
                raise RuntimeError(
                    f"Failed to write to snapshot {output_filename_prefix_iter}.state.json file, refusing to continue"
                )

        # wipe older snapshots
        if self.comm.rank == 0 and len(old_state_files) > 0:
            for old_file in old_state_files:
                print(f"Wiping old snapshot file {old_file}")
                Path(old_file).unlink()
                Path(
                    old_file.replace(".state.json", f".configs{config_suffix}")
                ).unlink()

    def read_rngs(self, bit_generator_states, different_nlocal=False):
        """read rngs from a dict

        Parameters
        ----------
        bit_generator_states: dict
            dict with "global" rng state and "locals" list of rng states
        different_nlocal: bool, default False
            if True, allow rngs from a different number of parallel processes than saved
            in dict (dropping extras or generating new ones, as needed)
        """
        if self.comm.rank == 0:
            if (
                len(bit_generator_states["locals"]) != self.comm.size
                and not different_nlocal
            ):
                raise RuntimeError(
                    f"Got local rngs states for {len(bit_generator_states['locals'])} procs, but current number is {self.comm.size}"
                )
        else:
            bit_generator_states = {"global": None, "locals": []}

        # bcast globals
        new_state = self.comm.bcast(bit_generator_states["global"], root=0)
        self.rng_global.bit_generator.state = new_state

        # scatter or generate locals
        n_locals = self.comm.bcast(len(bit_generator_states["locals"]), root=0)
        if n_locals == self.comm.size:
            self.rng_local.bit_generator.state = self.comm.scatter(
                bit_generator_states["locals"][: self.comm.size], root=0
            )
        else:
            # number of read-in generators doesn't match number needed, start over by generating all from global rng
            self.rng_global, self.rng_local = new_rngs(
                self.comm.rank, None, self.rng_global
            )

    def write_rngs(self):
        """Write rngs to a file"""
        if self.comm.rank == 0:
            bit_generator_states = {"global": self.rng_global.bit_generator.state}
            bit_generator_states["locals"] = self.comm.gather(
                self.rng_local.bit_generator.state, root=0
            )
        else:
            bit_generator_states = None
            _ = self.comm.gather(self.rng_local.bit_generator.state, root=0)

        return bit_generator_states
