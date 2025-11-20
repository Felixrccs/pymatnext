import torch

import numpy as np
import torch_sim as ts

from dataclasses import dataclass

from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState
#from torch_sim.neighbors import primitive_neighbor_list
import torch_sim.math as fm
from torch_sim import transforms

from ase import Atoms
from .atoms_contig_store import AtomsContiguousStorage


device = "cuda"
dtype = torch.float32


@dataclass(kw_only=True)
class WalkState(SimState):
    """State information for molecular dynamics simulations.

    This class represents the complete state of a molecular system being integrated
    with molecular dynamics. It extends the base SimState class to include additional
    attributes required for MD simulations, such as momenta, energy, and forces.
    The class also provides computed properties like velocities.

    Attributes:
        positions (torch.Tensor): Particle positions [n_particles, n_dim]
        masses (torch.Tensor): Particle masses [n_particles]
        cell (torch.Tensor): Simulation cell matrix [n_systems, n_dim, n_dim]
        pbc (bool): Whether to use periodic boundary conditions
        system_idx (torch.Tensor): System indices [n_particles]
        atomic_numbers (torch.Tensor): Atomic numbers [n_particles]
        momenta (torch.Tensor): Particle momenta [n_particles, n_dim]
        energy (torch.Tensor): Total energy of the system [n_systems]
        forces (torch.Tensor): Forces on particles [n_particles, n_dim]

    Properties:
        velocities (torch.Tensor): Particle velocities [n_particles, n_dim]
        n_systems (int): Number of independent systems in the batch
        device (torch.device): Device on which tensors are stored
        dtype (torch.dtype): Data type of tensors
    """

    energy: torch.Tensor
    tags: torch.Tensor

    _atom_attributes = SimState._atom_attributes | {"tags"}  # noqa: SLF001
    _system_attributes = SimState._system_attributes | {"energy"}  # noqa: SLF001


def atoms_to_state(
    atoms: "Atoms | list[Atoms]", device: torch.device, dtype: torch.dtype
) -> "ts.SimState":
    """Convert an ASE Atoms object or list of Atoms objects to a SimState.

    Args:
        atoms (Atoms | list[Atoms]): Single ASE Atoms object or list of Atoms objects
        device (torch.device): Device to create tensors on
        dtype (torch.dtype): Data type for tensors (typically torch.float32 or
            torch.float64)

    Returns:
        SimState: TorchSim SimState object.

    Raises:
        ImportError: If ASE is not installed
        ValueError: If systems have inconsistent periodic boundary conditions

    Notes:
        - Input positions and cell should be in Å
        - Input masses should be in amu
        - All systems must have consistent periodic boundary conditions
    """
    try:
        from ase import Atoms
    except ImportError:
        raise ImportError("ASE is required for atoms_to_state conversion") from None

    atoms_list = [atoms] if isinstance(atoms, Atoms) else atoms

    # Stack all properties in one go
    positions = torch.tensor(
        np.concatenate([at.positions for at in atoms_list]), dtype=dtype, device=device
    )
    masses = torch.tensor(
        np.concatenate([at.get_masses() for at in atoms_list]),
        dtype=dtype,
        device=device,
    )
    atomic_numbers = torch.tensor(
        np.concatenate([at.get_atomic_numbers() for at in atoms_list]),
        dtype=torch.int,
        device=device,
    )
    tags = torch.tensor(
        np.concatenate([at.get_tags() for at in atoms_list]),
        dtype=torch.int,
        device=device,
    )

    energy = torch.tensor(
        np.array([at.info["NS_energy"] for at in atoms_list]),
        dtype=dtype,
        device=device,
    )

    cell = torch.tensor(  # Transpose cell from ASE convention to TorchSim convention
        np.stack([at.cell.array.T for at in atoms_list]), dtype=dtype, device=device
    )

    # Create system indices using repeat_interleave
    atoms_per_system = torch.tensor([len(at) for at in atoms_list], device=device)
    system_idx = torch.repeat_interleave(
        torch.arange(len(atoms_list), device=device), atoms_per_system
    )

    # Verify consistent pbc
    if not all(all(at.pbc) == all(atoms_list[0].pbc) for at in atoms_list):
        raise ValueError("All systems must have the same periodic boundary conditions")

    return WalkState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=all(atoms_list[0].pbc),
        atomic_numbers=atomic_numbers,
        system_idx=system_idx,
        tags=tags,
        energy=energy,
    )


def state_to_atoms(state: "WalkState") -> list["Atoms"]:
    """Convert a SimState to a list of ASE Atoms objects.

    Args:
        state (SimState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[Atoms]: ASE Atoms objects, one per system

    Raises:
        ImportError: If ASE is not installed

    Notes:
        - Output positions and cell will be in Å
        - Output masses will be in amu
    """
    try:
        from ase import Atoms
        from ase.data import chemical_symbols
    except ImportError:
        raise ImportError("ASE is required for state_to_atoms conversion") from None

    # Convert tensors to numpy arrays on CPU
    positions = state.positions.detach().cpu().numpy()
    cell = state.cell.detach().cpu().numpy()  # Shape: (n_systems, 3, 3)
    atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
    system_indices = state.system_idx.detach().cpu().numpy()
    tags = state.tags.detach().cpu().numpy()
    energy = state.energy.detach().cpu().numpy()

    atoms_list = []
    for sys_idx in np.unique(system_indices):
        mask = system_indices == sys_idx
        system_positions = positions[mask]
        system_numbers = atomic_numbers[mask]
        system_tags = tags[mask]
        system_cell = cell[sys_idx].T  # Transpose for ASE convention

        # Convert atomic numbers to chemical symbols
        symbols = [chemical_symbols[z] for z in system_numbers]

        atoms = AtomsContiguousStorage(
            symbols=symbols,
            positions=system_positions,
            cell=system_cell,
            pbc=state.pbc,
            tags=system_tags,
        )
        atoms.info["NS_energy"] = energy[sys_idx]
        atoms_list.append(atoms)

    return atoms_list


def append_state_to_atoms(state: "WalkState", atoms: list["Atoms"]) -> list["Atoms"]:
    """Convert a SimState to a list of ASE Atoms objects.

    Args:
        state (SimState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[Atoms]: ASE Atoms objects, one per system

    Raises:
        ImportError: If ASE is not installed

    Notes:
        - Output positions and cell will be in Å
        - Output masses will be in amu
    """
    try:
        from ase import Atoms
        from ase.data import chemical_symbols
    except ImportError:
        raise ImportError("ASE is required for state_to_atoms conversion") from None

    # Convert tensors to numpy arrays on CPU
    positions = state.positions.detach().cpu().numpy()
    atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
    system_indices = state.system_idx.detach().cpu().numpy()
    energy = state.energy.detach().cpu().numpy()

    atoms_list = []
    for sys_idx in np.unique(system_indices):
        mask = system_indices == sys_idx
        system_positions = positions[mask]
        system_numbers = atomic_numbers[mask]

        # Convert atomic numbers to chemical symbols
        symbols = [chemical_symbols[z] for z in system_numbers]

        atoms[sys_idx].set_positions(system_positions)
        atoms[sys_idx].set_chemical_symbols(symbols)
        atoms[sys_idx].info["NS_energy"] = energy[sys_idx]
#        atoms_list.append(atoms)
#
#    return atoms_list


def state_init(
    atoms: "Atoms | list[Atoms]",
    model: ModelInterface,
) -> WalkState:
    """Initialize a swap Monte Carlo state from input data.

    Creates an initial state for swap Monte Carlo simulations by computing initial
    energy and setting up the permutation tracking. The simulation uses the Metropolis
    criterion to accept or reject proposed swaps based on energy differences.

    Make sure that if the trajectory is being reported, the
    `TorchSimTrajectory.write_state` method is called with `variable_masses=True`.

    Args:
        model: Energy model that takes a SimState and returns a dict containing
            'energy' as a key
        state: The simulation state to initialize from

    Returns:
        SwapMCState: Initialized state for swap Monte Carlo simulation containing
            positions, energy, and permutation tracking

    Examples:
        >>> mc_state = swap_monte_carlo_init(model=energy_model, state=initial_state)
        >>> for _ in range(100):
        >>>     mc_state = swap_monte_carlo_step(model, mc_state, kT=0.1)
    """
    state = ts.initialize_state(atoms, dtype=model.dtype, device=model.device)
    tags = torch.tensor(
        np.concatenate([at.get_tags() for at in atoms]),
        dtype=torch.int,
        device=model.device,
    )

    model_output = model(state)

    return WalkState(
        positions=state.positions,
        masses=state.masses,
        cell=state.cell,
        pbc=state.pbc,
        atomic_numbers=state.atomic_numbers,
        system_idx=state.system_idx,
        energy=model_output["energy"],
        tags=tags,
    )


class torch_walker:
    def __init__(self, lower, upper, model, E_model, atoms):
        self.model = model
        self.E_model = E_model
        self.Emax = torch.tensor(
            [0.0], dtype=self.model.dtype, device=self.model.device
        )
        self.n_atoms = len(atoms)
        self.n_tags = int(sum(atoms.get_tags()))
        cell = torch.tensor(atoms.cell.array.T, device=self.model.device, dtype=self.model.dtype)
        self.xy_shift = self.get_xy_shift(cell)
        self.xyz_shift = self.get_xyz_shift(cell)
        self.lower = torch.tensor(
            [lower], dtype=self.model.dtype, device=self.model.device
        )
        self.upper = torch.tensor(
            [upper], dtype=self.model.dtype, device=self.model.device
        )
        self.cutoff = torch.tensor([0.9,1.3,1.7], dtype=self.model.dtype, device=self.model.device)
        self.cutoff_numbers = torch.tensor([[8,8],[8,29],[29,29]],dtype=torch.int32, device=self.model.device)


    def get_xy_shift(self, cell):
        lattice = 4
        xy = torch.cartesian_prod(torch.arange(0, 1, 1/lattice, dtype=self.model.dtype, device=self.model.device), torch.arange(0, 1, 1/lattice, dtype=self.model.dtype, device=self.model.device))
        shifts = torch.cat([xy, torch.zeros((lattice*lattice,1),dtype=self.model.dtype, device=self.model.device)],1)
        return torch.matmul(shifts, cell)

    def get_xyz_shift(self, cell):
        lattice = 4
        shifts = torch.cartesian_prod(
            torch.arange(0, 1, 1/lattice, dtype=self.model.dtype, device=self.model.device),
            torch.arange(0, 1, 1/lattice, dtype=self.model.dtype, device=self.model.device),
            torch.tensor([-1, 1], dtype=self.model.dtype, device=self.model.device)
        )
        cell[2] *= 1.816/ torch.linalg.norm(cell[2])
        return torch.matmul(shifts, cell)

    def neighbor_collision(self,positions,numbers, cell):
        nl = torch_neighbor_list('d',[True,True,True], cell=cell, positions=positions,numbers=numbers, cutoff=self.cutoff, cutoff_numbers=self.cutoff_numbers, dtype=self.model.dtype, device=self.model.device)
        return nl < 1




    def walk_pos_gmc(self, state, steps, step_size):
        positions = state.positions.clone()

        torch.zeros(
            state.energy.shape, device=self.model.device, dtype=self.model.dtype
        )
        n_failed_in_a_row = 0
        velocities = torch.einsum(
            "ij,i->ij",
            torch.normal(
                0.0,
                step_size,
                size=positions.shape,
                device=self.model.device,
                dtype=self.model.dtype,
            ),
            state.tags,
        )
        for i_step in range(steps):
            # GMC step
            positions = positions.add(velocities)
            reflect_v_z(positions=positions, velocities=velocities, tags=state.tags, lower=self.lower, upper=self.upper)
            results = self.model(
                dict(
                    positions=positions,
                    cell=state.cell,
                    atomic_numbers=state.atomic_numbers,
                    system_idx=state.system_idx,
                    pbc=True,
                )
            )
            E = results["energy"]
            mask = (E >= self.Emax).to(torch.int32)

            F = torch.einsum("ij,i->ij", results["forces"], state.tags)
            F = F.reshape((state.n_systems, self.n_atoms, 3))
            velocities = velocities.reshape(F.shape)

            Norm = torch.einsum("ijk,ijk->i", F, F)
            F_hat = torch.einsum("lik,l->lik", F, 1 / torch.sqrt(Norm))
            projection = 2 * torch.einsum(
                "ijk,imn,imn,i->ijk", F_hat, velocities, F_hat, mask
            )
            velocities = torch.reshape(velocities - projection, positions.shape)

            n_failed_in_a_row += mask
            n_failed_in_a_row = torch.where(
                n_failed_in_a_row < 2, n_failed_in_a_row * mask, mask
            )

        successful = torch.logical_and(n_failed_in_a_row == 0, results["collision"])
        new_positions = state.positions.clone()
        for idx, val in enumerate(successful):
            if val:
                mask = state.system_idx == idx
                new_positions[mask] = positions[mask]
                state.energy[idx] = E[idx]
        state.positions = new_positions

        return [("gmc", len(successful), int(torch.sum(successful.to(torch.int32))))]

    def random_pos(self, state):
        positions = state.positions.clone()
        idx = batch_random_id(self.n_atoms, state.n_systems, self.n_tags, state.tags, device=self.model.device)
        positions[idx] = torch.matmul(torch.rand((state.n_systems,3),dtype=self.model.dtype, device=self.model.device), self.cell) + torch.tensor([[0.,0.,self.lower]]*state.n_systems,dtype=self.model.dtype, device=self.model.device)
        
        results = self.E_model(
            dict(
                positions=positions,
                cell=state.cell,
                atomic_numbers=state.atomic_numbers,
                system_idx=state.system_idx,
                pbc=True,
            )
        )
        E = results["energy"]

        new_positions = state.positions.clone()
        successful = torch.logical_and(E <= self.Emax, results["collision"])
        for idx, val in enumerate(successful):
            if val:
                mask = state.system_idx == idx
                new_positions[mask] = positions[mask]
                state.energy[idx] = E[idx]
        state.positions = new_positions


    def side_step(self, state):
        positions = state.positions.clone()
        lattice = 4
        idx = batch_random_id(self.n_atoms, state.n_systems, self.n_tags, state.tags, device=self.model.device)
        positions[idx] += self.xy_shift[torch.randint(1,16,(state.n_systems,))]

        results = self.E_model(
            dict(
                positions=positions,
                cell=state.cell,
                atomic_numbers=state.atomic_numbers,
                system_idx=state.system_idx,
                pbc=True,
            )
        )
        E = results["energy"]

        new_positions = state.positions.clone()
        successful = torch.logical_and(E <= self.Emax, results["collision"])
        for idx, val in enumerate(successful):
            if val:
                mask = state.system_idx == idx
                new_positions[mask] = positions[mask]
                state.energy[idx] = E[idx]
        state.positions = new_positions

    def up_and_down_step(self, state):
        positions = state.positions.clone()
        lattice = 4
        idx = batch_random_id(self.n_atoms, state.n_systems, self.n_tags, state.tags, device=self.model.device)
        positions[idx] += self.xyz_shift[torch.randint(0,16*2,(state.n_systems,))]
        reflect_z(positions=positions, tags=state.tags, lower=self.lower, upper=self.upper)


        results = self.E_model(
            dict(
                positions=positions,
                cell=state.cell,
                atomic_numbers=state.atomic_numbers,
                system_idx=state.system_idx,
                pbc=True,
            )
        )
        E = results["energy"]

        new_positions = state.positions.clone()
        successful = torch.logical_and(E <= self.Emax, results["collision"])
        for idx, val in enumerate(successful):
            if val:
                mask = state.system_idx == idx
                new_positions[mask] = positions[mask]
                state.energy[idx] = E[idx]
        state.positions = new_positions


    def walk(self, atoms, walk_len, Emax, step_size):
        self.Emax = torch.tensor([Emax], dtype=self.model.dtype, device=self.model.device        )

        state = atoms_to_state(atoms, device=self.model.device, dtype=self.model.dtype)
        self.cell = state.cell[0].clone()
        self.cell[2] = torch.tensor([0,0,self.upper-self.lower], device=self.model.device, dtype=self.model.dtype)

        n_att_acc = {"gmc": np.array([0, 0])}
        walk_len_so_far = 0
        while walk_len_so_far < walk_len:

            # returns list of tuples with move param attempt/success statistics
            n_att_acc_walk = self.walk_pos_gmc(state, 10, step_size)
            for param, n_att, n_acc in n_att_acc_walk:
                n_att_acc[param] += (n_att, n_acc)

            walk_len_so_far += 10

        self.random_pos(state)
        self.side_step(state)
        self.up_and_down_step(state)

        append_state_to_atoms(state,atoms)

        return n_att_acc

@torch.jit.script
def batch_random_id(
    n_atoms: int,
    n_systems: int,
    n_tags: int,
    tags: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
    possible_idx = torch.arange(n_atoms*n_systems, dtype=dtype, device=device)[tags==1]
    idx = possible_idx[torch.randint(low=0, high=n_tags, size=(n_systems,), dtype=dtype, device=device)
            +torch.arange(0, n_tags*n_systems, n_tags, dtype=dtype, device=device)]
    return idx




@torch.jit.script
def reflect_v_z(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    tags: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    ) -> None:
    z = positions.select(1, 2)
    vz = velocities.select(1, 2)

    above = torch.logical_and(z > upper, tags)
    below = torch.logical_and(z < lower, tags)
    mask = above | below

    vz.mul_(mask.to(vz.dtype).mul_(-2).add_(1))  # in-place sign flip
    z.copy_(
        torch.where(
            above, 2 * upper - z, torch.where(below, 2 * lower - z, z)
        )
    )


@torch.jit.script
def reflect_z(
    positions: torch.Tensor,
    tags: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    ) -> None:
    z = positions.select(1, 2)

    above = torch.logical_and(z > upper, tags)
    below = torch.logical_and(z < lower, tags)

    z.copy_(
        torch.where(
            above, 2 * upper - z, torch.where(below, 2 * lower - z, z)
        )
    )







@torch.jit.script
def torch_neighbor_list(  # noqa: C901, PLR0915
    quantities: str,
    pbc: tuple[bool, bool, bool],
    cell: torch.Tensor,
    positions: torch.Tensor,
    numbers: torch.Tensor,
    cutoff: torch.Tensor,
    cutoff_numbers,
    device: torch.device,
    dtype: torch.dtype,
    self_interaction: bool = False,  # noqa: FBT001, FBT002
    use_scaled_positions: bool = False,  # noqa: FBT001, FBT002
    max_n_bins: int = int(1e6),
) -> torch.Tensor:
    """Compute a neighbor list for an atomic configuration.

    ASE periodic neighbor list implementation
    Atoms outside periodic boundaries are mapped into the unit cell. Atoms
    outside non-periodic boundaries are included in the neighbor list
    but complexity of neighbor list search for those can become n^2.
    The neighbor list is sorted by first atom index 'i', but not by second
    atom index 'j'.

    Args:
        quantities: Quantities to compute by the neighbor list algorithm. Each character
            in this string defines a quantity. They are returned in a tuple of
            the same order. Possible quantities are
                * 'i' : first atom index
                * 'j' : second atom index
                * 'd' : absolute distance
                * 'D' : distance vector
                * 'S' : shift vector (number of cell boundaries crossed by the bond
                  between atom i and j). With the shift vector S, the
                  distances D between atoms can be computed from:
                  D = positions[j]-positions[i]+S.dot(cell)
        pbc: 3-tuple indicating giving periodic boundaries in the three Cartesian
            directions.
        cell: Unit cell vectors according to the row vector convention, i.e.
            `[[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]`.
        positions: Atomic positions. Anything that can be converted to an ndarray of
            shape (n, 3) will do: [(x1,y1,z1), (x2,y2,z2), ...]. If
            use_scaled_positions is set to true, this must be scaled positions.
        cutoff: Cutoff for neighbor search. It can be:
            * A single float: This is a global cutoff for all elements.
            * A dictionary: This specifies cutoff values for element
              pairs. Specification accepts element numbers of symbols.
              Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
            * A list/array with a per atom value: This specifies the radius of
              an atomic sphere for each atoms. If spheres overlap, atoms are
              within each others neighborhood.
              See :func:`~ase.neighborlist.natural_cutoffs`
              for an example on how to get such a list.
        device: PyTorch device to use for computations
        dtype: PyTorch data type to use
        self_interaction: Return the atom itself as its own neighbor if set to true.
            Default: False
        use_scaled_positions: If set to true, positions are expected to be
            scaled positions.
        max_n_bins: Maximum number of bins used in neighbor search. This is used to limit
            the maximum amount of memory required by the neighbor list.

    Returns:
        list[torch.Tensor]: One tensor for each item in `quantities`. Indices in `i`
            are returned in ascending order 0..len(a)-1, but the order of (i,j)
            pairs is not guaranteed.

    References:
        - This code is modified version of the github gist
        https://gist.github.com/Linux-cpp-lisp/692018c74b3906b63529e60619f5a207
    """
    # Naming conventions: Suffixes indicate the dimension of an array. The
    # following convention is used here:
    # c: Cartesian index, can have values 0, 1, 2
    # i: Global atom index, can have values 0..len(a)-1
    # xyz: Bin index, three values identifying x-, y- and z-component of a
    #         spatial bin that is used to make neighbor search O(n)
    # b: Linearized version of the 'xyz' bin index
    # a: Bin-local atom index, i.e. index identifying an atom *within* a
    #     bin
    # p: Pair index, can have value 0 or 1
    # n: (Linear) neighbor index

    if len(positions) == 0:
        raise RuntimeError("No atoms provided")

    # Compute reciprocal lattice vectors.
    recip_cell = torch.linalg.pinv(cell).T
    b1_c, b2_c, b3_c = recip_cell[0], recip_cell[1], recip_cell[2]

    # Compute distances of cell faces.
    l1 = torch.linalg.norm(b1_c)
    l2 = torch.linalg.norm(b2_c)
    l3 = torch.linalg.norm(b3_c)
    pytorch_scalar_1 = torch.as_tensor(1.0, device=device, dtype=dtype)
    face_dist_c = torch.hstack(
        [
            1 / l1 if l1 > 0 else pytorch_scalar_1,
            1 / l2 if l2 > 0 else pytorch_scalar_1,
            1 / l3 if l3 > 0 else pytorch_scalar_1,
        ]
    )
    if face_dist_c.shape != (3,):
        raise ValueError(f"face_dist_c.shape={face_dist_c.shape} != (3,)")

    

    # We use a minimum bin size of 3 A
    bin_size = torch.tensor(3.0, device=device, dtype=dtype)
    # Compute number of bins such that a sphere of radius cutoff fits into
    # eight neighboring bins.
    n_bins_c = torch.maximum(
        (face_dist_c / bin_size).to(dtype=torch.long, device=device),
        torch.ones(3, dtype=torch.long, device=device),
    )
    n_bins = torch.prod(n_bins_c)
    # Make sure we limit the amount of memory used by the explicit bins.
    while n_bins > max_n_bins:
        n_bins_c = torch.maximum(
            n_bins_c // 2, torch.ones(3, dtype=torch.long, device=device)
        )
        n_bins = torch.prod(n_bins_c)

    # Compute over how many bins we need to loop in the neighbor list search.
    neigh_search = torch.ceil(bin_size * n_bins_c / face_dist_c).to(
        dtype=torch.long, device=device
    )
    neigh_search_x, neigh_search_y, neigh_search_z = (
        neigh_search[0],
        neigh_search[1],
        neigh_search[2],
    )

    # If we only have a single bin and the system is not periodic, then we
    # do not need to search neighboring bins
    pytorch_scalar_int_0 = torch.as_tensor(0, dtype=torch.long, device=device)
    neigh_search_x = (
        pytorch_scalar_int_0 if n_bins_c[0] == 1 and not pbc[0] else neigh_search_x
    )
    neigh_search_y = (
        pytorch_scalar_int_0 if n_bins_c[1] == 1 and not pbc[1] else neigh_search_y
    )
    neigh_search_z = (
        pytorch_scalar_int_0 if n_bins_c[2] == 1 and not pbc[2] else neigh_search_z
    )

    # Sort atoms into bins.
    if not any(pbc):
        scaled_positions_ic = positions
    elif use_scaled_positions:
        scaled_positions_ic = positions
        positions = torch.dot(scaled_positions_ic, cell)
    else:
        scaled_positions_ic = torch.linalg.solve(cell.T, positions.T).T

    bin_index_ic = torch.floor(scaled_positions_ic * n_bins_c).to(
        dtype=torch.long, device=device
    )
    cell_shift_ic = torch.zeros_like(bin_index_ic, device=device)

    for c in range(3):
        if pbc[c]:
            # (Note: torch.divmod does not exist in older numpy versions)
            cell_shift_ic[:, c], bin_index_ic[:, c] = fm.torch_divmod(
                bin_index_ic[:, c], n_bins_c[c]
            )
        else:
            bin_index_ic[:, c] = torch.clip(bin_index_ic[:, c], 0, n_bins_c[c] - 1)

    # Convert Cartesian bin index to unique scalar bin index.
    bin_index_i = bin_index_ic[:, 0] + n_bins_c[0] * (
        bin_index_ic[:, 1] + n_bins_c[1] * bin_index_ic[:, 2]
    )

    # atom_i contains atom index in new sort order.
    atom_i = torch.argsort(bin_index_i)
    bin_index_i = bin_index_i[atom_i]

    # Find max number of atoms per bin
    max_n_atoms_per_bin = torch.bincount(bin_index_i).max()

    # Sort atoms into bins: atoms_in_bin_ba contains for each bin (identified
    # by its scalar bin index) a list of atoms inside that bin. This list is
    # homogeneous, i.e. has the same size *max_n_atoms_per_bin* for all bins.
    # The list is padded with -1 values.
    atoms_in_bin_ba = -torch.ones(
        n_bins.item(), max_n_atoms_per_bin.item(), dtype=torch.long, device=device
    )
    for bin_cnt in range(int(max_n_atoms_per_bin.item())):
        # Create a mask array that identifies the first atom of each bin.
        mask = torch.cat(
            (
                torch.ones(1, dtype=torch.bool, device=device),
                bin_index_i[:-1] != bin_index_i[1:],
            ),
            dim=0,
        )
        # Assign all first atoms.
        atoms_in_bin_ba[bin_index_i[mask], bin_cnt] = atom_i[mask]

        # Remove atoms that we just sorted into atoms_in_bin_ba. The next
        # "first" atom will be the second and so on.
        mask = torch.logical_not(mask)
        atom_i = atom_i[mask]
        bin_index_i = bin_index_i[mask]

    # Make sure that all atoms have been sorted into bins.
    if len(atom_i) != 0:
        raise ValueError(f"len(atom_i)={len(atom_i)} != 0")
    if len(bin_index_i) != 0:
        raise ValueError(f"len(bin_index_i)={len(bin_index_i)} != 0")

    # Now we construct neighbor pairs by pairing up all atoms within a bin or
    # between bin and neighboring bin. atom_pairs_pn is a helper buffer that
    # contains all potential pairs of atoms between two bins, i.e. it is a list
    # of length max_n_atoms_per_bin**2.
    # atom_pairs_pn_np = np.indices(
    #     (max_n_atoms_per_bin, max_n_atoms_per_bin), dtype=int
    # ).reshape(2, -1)
    atom_pairs_pn = torch.cartesian_prod(
        torch.arange(max_n_atoms_per_bin, device=device),
        torch.arange(max_n_atoms_per_bin, device=device),
    )
    atom_pairs_pn = atom_pairs_pn.T.reshape(2, -1)

    # Initialized empty neighbor list buffers.
    first_at_neigh_tuple_nn = []
    second_at_neigh_tuple_nn = []
    cell_shift_vector_x_n = []
    cell_shift_vector_y_n = []
    cell_shift_vector_z_n = []

    # This is the main neighbor list search. We loop over neighboring bins and
    # then construct all possible pairs of atoms between two bins, assuming
    # that each bin contains exactly max_n_atoms_per_bin atoms. We then throw
    # out pairs involving pad atoms with atom index -1 below.
    binz_xyz, biny_xyz, binx_xyz = torch.meshgrid(
        torch.arange(n_bins_c[2], device=device),
        torch.arange(n_bins_c[1], device=device),
        torch.arange(n_bins_c[0], device=device),
        indexing="ij",
    )
    # The memory layout of binx_xyz, biny_xyz, binz_xyz is such that computing
    # the respective bin index leads to a linearly increasing consecutive list.
    # The following assert statement succeeds:
    #     b_b = (binx_xyz + n_bins_c[0] * (biny_xyz + n_bins_c[1] *
    #                                     binz_xyz)).ravel()
    #     assert (b_b == torch.arange(torch.prod(n_bins_c))).all()

    # First atoms in pair.
    _first_at_neigh_tuple_n = atoms_in_bin_ba[:, atom_pairs_pn[0]]
    for dz in range(-int(neigh_search_z.item()), int(neigh_search_z.item()) + 1):
        for dy in range(-int(neigh_search_y.item()), int(neigh_search_y.item()) + 1):
            for dx in range(-int(neigh_search_x.item()), int(neigh_search_x.item()) + 1):
                # Bin index of neighboring bin and shift vector.
                shiftx_xyz, neighbinx_xyz = fm.torch_divmod(binx_xyz + dx, n_bins_c[0])
                shifty_xyz, neighbiny_xyz = fm.torch_divmod(biny_xyz + dy, n_bins_c[1])
                shiftz_xyz, neighbinz_xyz = fm.torch_divmod(binz_xyz + dz, n_bins_c[2])
                neighbin_b = (
                    neighbinx_xyz
                    + n_bins_c[0] * (neighbiny_xyz + n_bins_c[1] * neighbinz_xyz)
                ).ravel()

                # Second atom in pair.
                _second_at_neigh_tuple_n = atoms_in_bin_ba[neighbin_b][
                    :, atom_pairs_pn[1]
                ]

                # this basically just tiles shiftx_xyz.reshape(-1, 1) n times
                _cell_shift_vector_x_n = shiftx_xyz.reshape(-1, 1).repeat(
                    (1, int(max_n_atoms_per_bin.item() ** 2))
                )

                _cell_shift_vector_y_n = shifty_xyz.reshape(-1, 1).repeat(
                    (1, int(max_n_atoms_per_bin.item() ** 2))
                )

                _cell_shift_vector_z_n = shiftz_xyz.reshape(-1, 1).repeat(
                    (1, int(max_n_atoms_per_bin.item() ** 2))
                )


                # We have created too many pairs because we assumed each bin
                # has exactly max_n_atoms_per_bin atoms. Remove all superfluous
                # pairs. Those are pairs that involve an atom with index -1.
                mask = torch.logical_and(
                    _first_at_neigh_tuple_n != -1, _second_at_neigh_tuple_n != -1
                )
                if mask.sum() > 0:
                    first_at_neigh_tuple_nn += [_first_at_neigh_tuple_n[mask]]
                    second_at_neigh_tuple_nn += [_second_at_neigh_tuple_n[mask]]
                    cell_shift_vector_x_n += [_cell_shift_vector_x_n[mask]]
                    cell_shift_vector_y_n += [_cell_shift_vector_y_n[mask]]
                    cell_shift_vector_z_n += [_cell_shift_vector_z_n[mask]]

    # Flatten overall neighbor list.
    first_at_neigh_tuple_n = torch.cat(first_at_neigh_tuple_nn)
    second_at_neigh_tuple_n = torch.cat(second_at_neigh_tuple_nn)
    cell_shift_vector_n = torch.vstack(
        [
            torch.cat(cell_shift_vector_x_n),
            torch.cat(cell_shift_vector_y_n),
            torch.cat(cell_shift_vector_z_n),
        ]
    ).T

    # Add global cell shift to shift vectors
    cell_shift_vector_n += (
        cell_shift_ic[first_at_neigh_tuple_n] - cell_shift_ic[second_at_neigh_tuple_n]
    )

    # Remove all self-pairs that do not cross the cell boundary.
    if not self_interaction:
        m = torch.logical_not(
            torch.logical_and(
                first_at_neigh_tuple_n == second_at_neigh_tuple_n,
                (cell_shift_vector_n == 0).all(dim=1),
            )
        )
        first_at_neigh_tuple_n = first_at_neigh_tuple_n[m]
        second_at_neigh_tuple_n = second_at_neigh_tuple_n[m]
        cell_shift_vector_n = cell_shift_vector_n[m]

    # For non-periodic directions, remove any bonds that cross the domain
    # boundary.
    for c in range(3):
        if not pbc[c]:
            m = cell_shift_vector_n[:, c] == 0
            first_at_neigh_tuple_n = first_at_neigh_tuple_n[m]
            second_at_neigh_tuple_n = second_at_neigh_tuple_n[m]
            cell_shift_vector_n = cell_shift_vector_n[m]

    # Sort neighbor list.
    bin_cnt = torch.argsort(first_at_neigh_tuple_n)
    first_at_neigh_tuple_n = first_at_neigh_tuple_n[bin_cnt]
    second_at_neigh_tuple_n = second_at_neigh_tuple_n[bin_cnt]
    cell_shift_vector_n = cell_shift_vector_n[bin_cnt]

    # Compute distance vectors.
    # TODO: Use .T?
    distance_vector_nc = (
        positions[second_at_neigh_tuple_n]
        - positions[first_at_neigh_tuple_n]
        + cell_shift_vector_n.to(cell.dtype).matmul(cell)
    )
    abs_distance_vector_n = torch.sqrt(
        torch.sum(distance_vector_nc * distance_vector_nc, dim=1)
    )

    # We have still created too many pairs. Only keep those with distance
    # smaller than max_cutoff.
    per_pair_cutoff_n = torch.zeros_like(abs_distance_vector_n)
    for i, c in enumerate(cutoff):
        mask = torch.logical_or(
            torch.logical_and(
                numbers[first_at_neigh_tuple_n] == cutoff_numbers[i,0],
                numbers[second_at_neigh_tuple_n] == cutoff_numbers[i,1]),
            torch.logical_and(
                numbers[first_at_neigh_tuple_n] == cutoff_numbers[i,1],
                numbers[second_at_neigh_tuple_n] == cutoff_numbers[i,0]))
        per_pair_cutoff_n[mask] = c




    mask = abs_distance_vector_n < per_pair_cutoff_n
    #first_at_neigh_tuple_n = first_at_neigh_tuple_n[mask]
    #second_at_neigh_tuple_n = second_at_neigh_tuple_n[mask]
    #cell_shift_vector_n = cell_shift_vector_n[mask]
    #distance_vector_nc = distance_vector_nc[mask]
    #abs_distance_vector_n = abs_distance_vector_n[mask]

    return mask.sum()
