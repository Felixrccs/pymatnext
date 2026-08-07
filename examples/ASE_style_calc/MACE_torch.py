import torch 
import torch_sim as ts
from torch_sim.models.mace import MaceModel
from torch.nn.utils.rnn import pad_sequence

from batch_nl import NeighbourList

torch.set_float32_matmul_precision('high')




def new_forward(self, state: ts.SimState | ts.typing.StateDict) -> dict[str, torch.Tensor]:  # noqa: C901
        """Compute energies, forces, and stresses for the given atomic systems.

        Processes the provided state information and computes energies, forces, and
        stresses using the underlying MACE model. Handles batched calculations for
        multiple systems and constructs the necessary neighbor lists.

        Args:
            state (SimState | StateDict): State object containing positions, cell,
                and other system information. Can be either a SimState object or a
                dictionary with the relevant fields.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - 'energy': System energies with shape [n_systems]
                - 'forces': Atomic forces with shape [n_atoms, 3] if compute_forces=True
                - 'stress': System stresses with shape [n_systems, 3, 3] if
                    compute_stress=True

        Raises:
            ValueError: If atomic numbers are not provided either in the constructor
                or in the forward pass, or if provided in both places.
            ValueError: If system indices are not provided when needed.
        """
        sim_state = (
            state
            if isinstance(state, ts.SimState)
            else ts.SimState(**state, masses=torch.ones_like(state["positions"]))
        )

        # Handle input validation for atomic numbers
        if sim_state.atomic_numbers is None and not self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers must be provided in either the constructor or forward."
            )
        if sim_state.atomic_numbers is not None and self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers cannot be provided in both the constructor and forward."
            )

        # Use system_idx from init if not provided
        if sim_state.system_idx is None:
            if not hasattr(self, "system_idx"):
                raise ValueError(
                    "System indices must be provided if not set during initialization"
                )
            sim_state.system_idx = self.system_idx

        # Update system_idx information if new atomic numbers are provided
        if (
            sim_state.atomic_numbers is not None
            and not self.atomic_numbers_in_init
            and not torch.equal(
                sim_state.atomic_numbers,
                getattr(self, "atomic_numbers", torch.zeros(0, device=self.device)),
            )
        ):
            self.setup_from_system_idx(sim_state.atomic_numbers, sim_state.system_idx)
        positions = sim_state.positions.clone() % torch.diagonal(sim_state.row_vector_cell[0]) # wrapping for orthoromic cell


        nl = NeighbourList(
            positions=positions,
            system_idx=sim_state.system_idx,
            cells=sim_state.row_vector_cell,
            cutoff=self.r_max,
            device=self.device,
        )
        nl.load_data()
        edge_index, unit_shifts, shifts, d, config_idx  = nl.calculate_neighbourlist(use_torch_compile=True)

        # Todo: check cutoff for collisions
        at_numbers = torch.sum(sim_state.atomic_numbers[edge_index].T,dim =1)
        i_j_cutoff = torch.zeros_like(at_numbers,dtype=torch.float32)
        i_j_cutoff[at_numbers == 58] = 1.7
        i_j_cutoff[at_numbers == 37] = 1.3
        i_j_cutoff[at_numbers == 16] = 0.9
        collision = d<i_j_cutoff
        sys_ids, split_len = torch.unique(config_idx, sorted=True, return_counts=True)
        sys_collisions = torch.ones(sim_state.n_systems, dtype=torch.bool)
        sys_collisions[sys_ids] = torch.sum(pad_sequence(
            torch.split(collision, tuple(split_len)),
            batch_first=True,
            padding_value=False,
        ), dim=1)== 0
        # _, split_len = torch.unique(config_idx, sorted=True, return_counts=True)
        # sys_collisions = torch.sum(pad_sequence(
        #     torch.split(collision, tuple(split_len)),
        #     batch_first=True,
        #     padding_value=False,
        # ), dim=1)== 0



        # Head for whole batch
        head = torch.ones(self.n_systems, device=self.device, dtype=torch.int32)


        # Get model output
        out = self.model(
            dict(
                ptr=self.ptr,
                node_attrs=self.node_attrs,
                batch=sim_state.system_idx,
                pbc=sim_state.pbc,
                cell=sim_state.row_vector_cell,
                positions=positions,
                edge_index=edge_index,
                unit_shifts=unit_shifts,
                shifts=shifts,
                head=head,
            ),
            compute_force=self.compute_forces,
            compute_stress=self.compute_stress,
        )
        results: dict[str, torch.Tensor] = {}

        # Process energy
        energy = out["energy"]
        if energy is not None:
            results["energy"] = energy.detach()
        else:
            results["energy"] = torch.zeros(self.n_systems, device=self.device)

        # Process forces
        if self.compute_forces:
            forces = out["forces"]
            if forces is not None:
                results["forces"] = forces.detach()

        # Process stress
        if self.compute_stress:
            stress = out["stress"]
            if stress is not None:
                results["stress"] = stress.detach()

        results["collision"] = sys_collisions

        return results



def new__init__(self,
    positions: torch.tensor,
    system_idx: torch.tensor,
    cells: torch.tensor,
    cutoff: float,
    float_dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
    ):
    """
    Initialize a batched neighbour-list calculator.
    Parameters
    ----------
    positions : torch.tensor
        All positions in a (N_config x N_atoms, 3) tensor
    system_idx: torch.tensor
        Map from position to config (N_config x N_atoms) tensor
    cells : torch.tensor (N_config, 3, 3)
    cutoff : float
        Cutoff radius used for neighbour detection. Must be positive.
    float_dtype: str or torch.dtype, optional
    device : str or torch.device, optional
        Device on which all batched tensors will be allocated
        ("cpu", "cuda", or torch.device(...)). If None, CUDA is used
        when available, otherwise CPU.
    """
    
    self.positions = positions
    self.batch_cell_tensor = cells
    self.system_idx = system_idx
    sys, self.len_sys = torch.unique(self.system_idx, sorted=True, return_counts=True)
    self.num_configs = len(sys)
    if self.num_configs != cells.shape[0]:
        raise ValueError(f"number of conifgs and cell lists should be the same, got num_configs = {self.num_configs} and len(cell_list) = {cells.shape[0]}")
    # dtype check 
    if float_dtype not in [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ]:
        raise TypeError(
            f"float_dtype must be a floating torch.dtype, got {float_dtype}."
        )
    self.float_dtype = float_dtype
    self.int_dtype = torch.long
    # cutoff
    self.cutoff = torch.as_tensor(cutoff, dtype=self.float_dtype)
    if self.cutoff.ndim != 0:
        raise ValueError(
            f"cutoff must be a scalar, got tensor with shape {tuple(self.cutoff.shape)}."
        )
    if self.cutoff.item() <= 0.0:
        raise ValueError(
            f"cutoff must be positive, got {self.cutoff.item()}."
        )
    # device
    if device is None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, (str, torch.device)):
        raise TypeError(
            "device should be a string or torch.device, "
            f"got {type(device).__name__}."
        )
    else:
        self.device = torch.device(device)
    # internal tolerance to filter self / near-self images
    self._tolerance = 1e-6
    # compiled neighbour-list function
    self._nlist_ON2_compiled = torch.compile(self._nlist_ON2)


def wrap_positions_triclinic_batched(batched_positions, batched_cells):
    """
    batched_positions: (B, N, 3) Cartesian coordinates
    batched_cells: (B, 3, 3) cell matrices (rows = lattice vectors)
    """
    H_inv = torch.inverse(batched_cells)                     # (B, 3, 3)

    # Cartesian → fractional
    frac = torch.matmul(batched_positions, H_inv.transpose(-1, -2))   # (B, N, 3)

    # Wrap fractional coords into [0.0, 1.0)
    frac_wrapped = frac % 1.0
    # Fractional → Cartesian
    batched_positions_wrapped = torch.matmul(frac_wrapped, batched_cells)         # (B, N, 3)
    return batched_positions_wrapped


def new_load_data(self):
        """
        Convert input positions and cells into padded batched tensors.

        This populates `batch_positions_tensor`, `batch_mask_tensor`,
        and `batch_cell_tensor`, and moves them to the configured device.

        Must be called before `calculate_neighbourlist`.
        """
        positions_list = torch.split(self.positions, tuple(self.len_sys)) #split of positions for padding

        self.batch_positions_tensor = pad_sequence(
            positions_list,
            batch_first=True,
            padding_value=float("nan"),
        )

        self.batch_mask_tensor = (self.batch_positions_tensor == self.batch_positions_tensor).any(dim=-1)

        self.batch_positions_tensor = torch.nan_to_num(self.batch_positions_tensor, nan=0)


def new_calculate_neighbourlist(self, use_torch_compile: bool = True):
    """
    Compute the batched neighbour list for all configurations.
    Notes
    -----
    `load_data()` must be called before invoking this method.
    Parameters
    ----------
    use_torch_compile : bool, optional
        If True, use the torch.compile-optimised backend; otherwise use
        the plain PyTorch implementation.
    Returns
    -------
    r_edges : torch.Tensor
        Tensor of shape (2, n_edges) with flattened source and neighbour
        atom indices in the global (batched) indexing.
    r_integer_lattice_shifts : torch.Tensor
        Tensor of shape (n_edges, 3) with integer lattice shift vectors.
    r_cartesian_lattice_shifts : torch.Tensor
        Tensor of shape (n_edges, 3) with Cartesian lattice shift vectors.
    r_distances : torch.Tensor
        Tensor of shape (n_edges,) with interatomic distances.
    """
    if use_torch_compile:
        neighbourlist_fn = self._nlist_ON2_compiled
    else:
        neighbourlist_fn = self._nlist_ON2
    # compute full batched neighbour data
    (
        distance_matrix,                       # (n_configs, n_lattice_shifts, n_max, n_max)
        criterion,                             # (n_configs, n_lattice_shifts, n_max, n_max)
        batch_lattice_shifts_tensor,           # (n_lattice_shifts, 3)
        batch_cartesian_lattice_shifts_tensor, # (n_configs, n_lattice_shifts, 3)
    ) = neighbourlist_fn(
        self.batch_positions_tensor,           # (n_configs, n_max, 3)
        self.batch_cell_tensor,                # (n_configs, 3, 3)
        self.batch_mask_tensor,                # (n_configs, n_max)
        self.cutoff,
        self._tolerance,
    )
    # indices of all neighbour pairs
    config_idx, lattice_shift_idx, atom_idx, neighbour_idx = torch.nonzero(
        criterion,
        as_tuple=True,
    )                                          # each (n_edges,)
    # global atom indices via per-config offsets
    lengths = self.batch_mask_tensor.sum(dim=-1, dtype=self.int_dtype)  # (n_configs,)
    offsets = torch.cumsum(lengths, dim=0) - lengths                    # (n_configs,)
    r_edges = torch.stack(
        [
            atom_idx      + offsets[config_idx],    # (n_edges,)
            neighbour_idx + offsets[config_idx],    # (n_edges,)
        ],
        dim=0,
    )                                               # (2, n_edges)
    r_integer_lattice_shifts = batch_lattice_shifts_tensor[lattice_shift_idx]          # (n_edges, 3)
    r_cartesian_lattice_shifts = batch_cartesian_lattice_shifts_tensor[
        config_idx,
        lattice_shift_idx,
    ]                                                                                  # (n_edges, 3)
    r_distances = distance_matrix[
        config_idx,
        lattice_shift_idx,
        atom_idx,
        neighbour_idx,
    ]                                                                                  # (n_edges,)
    return (
        r_edges,
        r_integer_lattice_shifts,
        r_cartesian_lattice_shifts,
        r_distances,
        config_idx,
    )


NeighbourList.__init__ = new__init__
NeighbourList.load_data = new_load_data
NeighbourList.calculate_neighbourlist = new_calculate_neighbourlist
MaceModel.forward = new_forward

device = "cuda"
dtype = torch.float64

mace = torch.load("./mace_6.model", map_location=device)
batched_energy = MaceModel(model=mace, device=device, dtype=dtype, compute_stress=False, compute_forces=False)
batched_forces = MaceModel(model=mace, device=device, dtype=dtype, compute_stress=False, compute_forces=True)

calc = [batched_energy, batched_forces]
