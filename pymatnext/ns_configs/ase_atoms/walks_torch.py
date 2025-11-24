import torch

import numpy as np
import torch_sim as ts

from dataclasses import dataclass

from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState

# from torch_sim.neighbors import primitive_neighbor_list
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
    """Class for a torch based monte carlo walk for nested sampling simulations.

    Attributes:
        model (ModelInterface): Batched torchsim model that return energies and forces.
        E_model (ModelInterface): Batched torchsim model that returns energies.
        n_atoms (int): The number of moving atoms per structure.
        tags (int): The number of moving atoms.
        xy_shift (torch.Tensor): Possible side steps on the lattice.
        xyz_shift (torch.Tensor): Possible up and down steps the lattice.
        lower (torch.Tensor): Lower boundary in z direction.
        upper (torch.Tensor): Upper boundary in z direction.

    """

    model: ModelInterface
    E_model: ModelInterface
    Emax: torch.Tensor
    n_atoms: int
    tags: int
    xy_shift: torch.Tensor
    xyz_shift: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor

    def __init__(
        self,
        z_limits: list,
        model: ModelInterface,
        E_model: ModelInterface,
        atoms: Atoms,
    )->None:
        """Setting of attributes

        Args:
            z_limits (list): lower and upper boundary in z direction
            model (ModelInterface): Batched torchsim model that return energies and forces
            E_model (ModelInterface): Batched torchsim model that returns energies
            atoms (Atoms): ase atoms or meta data extraction
        """
        lattice = 4
        self.model = model
        self.E_model = E_model
        self.Emax = torch.tensor(
            [0.0], dtype=self.model.dtype, device=self.model.device
        )
        self.n_atoms = len(atoms)
        self.n_tags = int(sum(atoms.get_tags()))
        # Todo check in cell math is correct (shouldn't matter in orthogonal systems)
        cell = torch.tensor(
            atoms.cell.array.T, device=self.model.device, dtype=self.model.dtype
        )
        self.xy_shift = self.get_xy_shift(cell, lattice)
        self.xyz_shift = self.get_xyz_shift(cell, lattice)
        self.lower = torch.tensor(
            [z_limits[0] * np.linalg.norm(atoms.cell[2])],
            dtype=self.model.dtype,
            device=self.model.device,
        )
        self.upper = torch.tensor(
            [z_limits[1] * np.linalg.norm(atoms.cell[2])],
            dtype=self.model.dtype,
            device=self.model.device,
        )

    def get_xy_shift(
        self,
        cell: torch.Tensor,
        lattice: int,
    ) -> torch.Tensor:
        """Setting up all possible side steps on the lattice

        Args:
            cell (torch.Tensor): Torchsim cell
            lattice (int): lattice size

        Returns:
            torch.Tensor: all possible shifts
        """
        xy = torch.cartesian_prod(
            torch.arange(
                0, 1, 1 / lattice, dtype=self.model.dtype, device=self.model.device
            ),
            torch.arange(
                0, 1, 1 / lattice, dtype=self.model.dtype, device=self.model.device
            ),
        )
        shifts = torch.cat(
            [
                xy,
                torch.zeros(
                    (lattice * lattice, 1),
                    dtype=self.model.dtype,
                    device=self.model.device,
                ),
            ],
            1,
        )
        return torch.matmul(shifts, cell)

    def get_xyz_shift(
        self,
        cell: torch.Tensor,
        lattice: int,
    ) -> torch.Tensor:
        """Setting up all possible up and down steps on the lattice

        Args:
            cell (torch.Tensor): Torchsim cell
            lattice (int): lattice size

        Returns:
            torch.Tensor: all possible shifts
        """
        shifts = torch.cartesian_prod(
            torch.arange(
                0, 1, 1 / lattice, dtype=self.model.dtype, device=self.model.device
            ),
            torch.arange(
                0, 1, 1 / lattice, dtype=self.model.dtype, device=self.model.device
            ),
            torch.tensor([-1, 1], dtype=self.model.dtype, device=self.model.device),
        )
        cell[2] *= 1.816 / torch.linalg.norm(cell[2])
        return torch.matmul(shifts, cell)

    def walk_pos_gmc(
        self,
        state: WalkState,
        steps: int,
        step_size: float,
    ) -> list:
        """Galilean Monte Carlo walk over n steps. Accepts walks in dependance on the energy and hard spheres

        Args:
            state (WalkState): Torchsim like state of batched atoms
            steps (int): number of steps
            step_size (float): standard deviation of normal distribution

        Returns:
            list: (move, accepted, total)
        """
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
            reflect_v_z(
                positions=positions,
                velocities=velocities,
                tags=state.tags,
                lower=self.lower,
                upper=self.upper,
            )
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

    def random_pos(
        self,
        state: WalkState,
    )->None:
        """Walks a single atom by a random position for each batched systeme

        Args:
            state (WalkState): Torchsim like state of batched atoms
        """
        positions = state.positions.clone()
        idx = batch_random_id(
            self.n_atoms,
            state.n_systems,
            self.n_tags,
            state.tags,
            device=self.model.device,
        )
        positions[idx] = torch.matmul(
            torch.rand(
                (state.n_systems, 3), dtype=self.model.dtype, device=self.model.device
            ),
            self.cell,
        ) + torch.tensor(
            [[0.0, 0.0, self.lower]] * state.n_systems,
            dtype=self.model.dtype,
            device=self.model.device,
        )

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

    def side_step(
        self,
        state: WalkState,
    )->None:
        """Walks a single atom on the lattice position for each batched systeme

        Args:
            state (WalkState): Torchsim like state of batched atoms
        """
        positions = state.positions.clone()
        idx = batch_random_id(
            self.n_atoms,
            state.n_systems,
            self.n_tags,
            state.tags,
            device=self.model.device,
        )
        positions[idx] += self.xy_shift[torch.randint(1, 16, (state.n_systems,))]

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

    def up_and_down_step(
        self,
        state: WalkState,
    )->None:
        """Walks a single atom up or down a step for each batched systeme

        Args:
            state (WalkState): Torchsim like state of batched atoms
        """
        positions = state.positions.clone()
        idx = batch_random_id(
            self.n_atoms,
            state.n_systems,
            self.n_tags,
            state.tags,
            device=self.model.device,
        )
        positions[idx] += self.xyz_shift[torch.randint(0, 16 * 2, (state.n_systems,))]
        reflect_z(
            positions=positions, tags=state.tags, lower=self.lower, upper=self.upper
        )

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

    def walk(
        self,
        atoms: list["Atoms"],
        walk_len: int,
        Emax: float,
        step_size: float,
    ) -> dict:
        """Convertion of ase.Atoms into TorchSim - WalkState. Combined MC walk and update of ase.Atoms.

        Args:
            atoms (list["Atoms"]): Atoms to walk
            walk_len (int): total walk length
            Emax (float): energy of highest walker/ phase space boundary for walk
            step_size (float): standard deviation for displacements in GMC walk

        Returns:
            dict: Accepted vs total steps
        """
        self.Emax = torch.tensor(
            [Emax], dtype=self.model.dtype, device=self.model.device
        )

        state = atoms_to_state(atoms, device=self.model.device, dtype=self.model.dtype)
        self.cell = state.cell[0].clone()
        self.cell[2] = torch.tensor(
            [0, 0, self.upper - self.lower],
            device=self.model.device,
            dtype=self.model.dtype,
        )

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

        append_state_to_atoms(state, atoms)

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
    possible_idx = torch.arange(n_atoms * n_systems, dtype=dtype, device=device)[
        tags == 1
    ]
    idx = possible_idx[
        torch.randint(low=0, high=n_tags, size=(n_systems,), dtype=dtype, device=device)
        + torch.arange(0, n_tags * n_systems, n_tags, dtype=dtype, device=device)
    ]
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
    z.copy_(torch.where(above, 2 * upper - z, torch.where(below, 2 * lower - z, z)))


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

    z.copy_(torch.where(above, 2 * upper - z, torch.where(below, 2 * lower - z, z)))
