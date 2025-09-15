import warnings, sys

from ase.calculators.calculator import all_changes
from ase.neighborlist import neighbor_list

import numpy as np
from copy import copy

one_third = 1.0 / 3.0


def walk_pos_gmc_old(ns_atoms, Emax, rng, move):
    """Walk atomic positions using Galilean Monte-Carlo

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        initial atomic configuration
    Emax: float
        maximum shifted energy
    rng: numpy.Generator
        random number generator

    Returns
    -------
    [(move, int n_attempt, int n_success)] info on move params and attempts/successes
    """
    atoms = ns_atoms.atoms
    # new random velocities
    atoms.arrays["NS_velocities"][...] = rng.normal(
        scale=ns_atoms.step_size[move], size=atoms.positions.shape
    )

    # below here operate only on _internal_ energy, without any "+ P V - mu N" shifts
    Emax -= atoms.info["NS_energy_shift"]

    # store orig position in case move is rejected
    atoms.prev_positions[...] = atoms.positions
    n_failed_in_a_row = 0

    # consider fixed atoms
    tags = atoms.get_tags()
    tags = np.where(tags >= ns_atoms.tags[move], 1, 0)
    moving = np.broadcast_to(tags[:, None], (len(atoms), 3))
    for i_step in range(ns_atoms.walk_traj_len[move]):
        # step and evaluate new energy, forces
        atoms.positions += atoms.arrays["NS_velocities"] * moving
        # inclusion of limits by instering reflective walls at the lower and the higher limits
        if ns_atoms.limit:
            tmp = atoms.get_scaled_positions(wrap=False)
            for k, val in enumerate(["x", "y", "z"]):
                if val in ns_atoms.limit:
                    lower_limit, upper_limit = ns_atoms.limit[val]

                    # Identify out-of-bounds indices
                    below_lower = tmp[:, k] <= lower_limit
                    above_upper = tmp[:, k] >= upper_limit

                    if np.any(below_lower) or np.any(above_upper):
                        # Mirror of velocities
                        change = np.ones_like(tmp[:, k])
                        change[below_lower] = -1
                        change[above_upper] = -1
                        atoms.arrays["NS_velocities"][:, k] *= change

                        # Mirror of positions
                        while np.any(below_lower) or np.any(above_upper):
                            tmp[below_lower, k] = 2 * lower_limit - tmp[below_lower, k]
                            tmp[above_upper, k] = 2 * upper_limit - tmp[above_upper, k]

                            # Update out-of-bounds indices
                            below_lower = tmp[:, k] <= lower_limit
                            above_upper = tmp[:, k] >= upper_limit

                        atoms.set_scaled_positions(
                            atoms.get_scaled_positions(wrap=False) * (1.0 - moving)
                            + tmp * moving
                        )

        atoms.calc.calculate(
            atoms, properties=["free_energy", "forces"], system_changes=all_changes
        )
        E = atoms.calc.results.get("free_energy", atoms.calc.results.get("energy"))
        F = atoms.calc.results["forces"]

        if E >= Emax:  # reflect or fail
            n_failed_in_a_row += 1
            if n_failed_in_a_row >= 2:
                break
            if np.sum(F * F) == 0:
                warnings.warn("Got F=0 while reflecting, giving up")
                break
            F_hat = F / np.sqrt(np.sum(F * F))
            atoms.arrays["NS_velocities"] -= (
                F_hat * 2.0 * np.sum(atoms.arrays["NS_velocities"] * F_hat)
            )
        else:
            n_failed_in_a_row = 0

    if n_failed_in_a_row > 0:
        # revert
        atoms.positions[...] = atoms.prev_positions

        return [(move, 1, 0)]
    else:
        atoms.info["NS_energy"][...] = E
        atoms.arrays["NS_forces"][...] = F

        return [(move, 1, 1)]


def walk_pos_gmc(ns_atoms, Emax, rng, move):
    """Walk atomic positions using Galilean Monte-Carlo

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        initial atomic configuration
    Emax: float
        maximum shifted energy
    rng: numpy.Generator
        random number generator

    Returns
    -------
    [(move, int n_attempt, int n_success)] info on move params and attempts/successes
    """
    atoms = ns_atoms.atoms
    tags = atoms.get_tags()
    # new random velocities

    atoms.arrays["NS_velocities"][...] = rng.normal(
        scale=ns_atoms.step_size[move], size=atoms.positions.shape
    )

    if 1 in tags and ns_atoms.step_size[move] > ns_atoms.one_max_step:
        one_tag = np.where(tags <= 1)[0]
        tmp = atoms.arrays["NS_velocities"][...]
        tmp[one_tag] = rng.normal(scale=ns_atoms.one_max_step, size=(len(one_tag), 3))
        atoms.arrays["NS_velocities"][...] = tmp
    else:
        tags = np.where(tags >= ns_atoms.tags[move], 1, 0)


    # below here operate only on _internal_ energy, without any "+ P V - mu N" shifts
    Emax -= atoms.info["NS_energy_shift"]

    # store orig position in case move is rejected
    atoms.prev_positions[...] = atoms.positions
    n_failed_in_a_row = 0

    # consider fixed atoms
    moving = np.where(tags >= ns_atoms.tags[move], 1, 0)
    moving = np.broadcast_to(tags[:, None], (len(atoms), 3))
    for i_step in range(ns_atoms.walk_traj_len[move]):
        # step and evaluate new energy, forces
        atoms.positions += atoms.arrays["NS_velocities"] * moving
        # inclusion of limits by instering reflective walls at the lower and the higher limits
        if ns_atoms.limit:
            tmp = atoms.get_scaled_positions(wrap=False)
            for k, val in enumerate(["x", "y", "z"]):
                if val in ns_atoms.limit:
                    lower_limit, upper_limit = ns_atoms.limit[val]

                    # Identify out-of-bounds indices
                    below_lower = tmp[:, k] <= lower_limit
                    above_upper = tmp[:, k] >= upper_limit

                    if np.any(below_lower) or np.any(above_upper):
                        # Mirror of velocities
                        change = np.ones_like(tmp[:, k])
                        change[below_lower] = -1
                        change[above_upper] = -1
                        atoms.arrays["NS_velocities"][:, k] *= change

                        # Mirror of positions
                        while np.any(below_lower) or np.any(above_upper):
                            tmp[below_lower, k] = 2 * lower_limit - tmp[below_lower, k]
                            tmp[above_upper, k] = 2 * upper_limit - tmp[above_upper, k]

                            # Update out-of-bounds indices
                            below_lower = tmp[:, k] <= lower_limit
                            above_upper = tmp[:, k] >= upper_limit

                        atoms.set_scaled_positions(
                            atoms.get_scaled_positions(wrap=False) * (1.0 - moving)
                            + tmp * moving
                        )
        try:
            atoms.calc.calculate(
                atoms, properties=["free_energy", "forces"], system_changes=all_changes
            )
        except:
            print(len(atoms))
            print(atoms.positions[...])
            sys.exit()
        E = atoms.calc.results.get("free_energy", atoms.calc.results.get("energy"))
        F = atoms.calc.results["forces"]

        if E >= Emax:  # reflect or fail
            n_failed_in_a_row += 1
            if n_failed_in_a_row >= 2:
                break
            if np.sum(F * F) == 0:
                warnings.warn("Got F=0 while reflecting, giving up")
                break
            new = np.zeros_like(atoms.arrays["NS_velocities"])
            for tag in np.unique(tags):
                if tag > 0:
                    tag_id = np.where(tags == tag)[0]
                    if np.sum(F[tag_id] * F[tag_id]) == 0:
                        warnings.warn("Got F=0 while reflecting, giving up")
                        break
                    F_hat = F[tag_id] / np.sqrt(np.sum(F[tag_id] * F[tag_id]))
                    new[tag_id] = atoms.arrays["NS_velocities"][tag_id] -  (
                        F_hat * 2.0 * np.sum(atoms.arrays["NS_velocities"][tag_id] * F_hat)
                    )
            atoms.arrays["NS_velocities"] = new
        else:
            n_failed_in_a_row = 0

    if n_failed_in_a_row > 0:
        # revert
        atoms.positions[...] = atoms.prev_positions

        return [(move, 1, 0)]
    else:
        atoms.info["NS_energy"][...] = E
        atoms.arrays["NS_forces"][...] = F

        return [(move, 1, 1)]


def walk_lattice_single(ns_atoms, Emax, rng, move):
    """Walk single atomic positions by lattice parameter in x and y direction

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        initial atomic configuration
    Emax: float
        maximum shifted energy
    rng: numpy.Generator
        random number generator
    move: str
        id of move

    Returns
    -------
    [] info on move params and attempts/successes
    """
    atoms = ns_atoms.atoms
    atoms.prev_positions[...] = atoms.positions
    atom_id = rng.choice(np.where(atoms.get_tags() >= ns_atoms.tags[move])[0])
    shift = np.array(
        [
            rng.integers(0, ns_atoms.lattice[0]) / ns_atoms.lattice[0],
            rng.integers(0, ns_atoms.lattice[1]) / ns_atoms.lattice[1],
            0,
        ]
    )
    tmp = atoms.get_scaled_positions(wrap=False)
    tmp[atom_id] += shift
    atoms.set_scaled_positions(tmp)

    d = neighbor_list(
        "d",
        atoms,
        cutoff=ns_atoms.min_dist,
        self_interaction=False,
    )

    if len(d) > 0:
        atoms.positions[...] = atoms.prev_positions
        return []

    atoms.calc.calculate(
        atoms, properties=["free_energy", "forces"], system_changes=all_changes
    )
    E = atoms.calc.results.get("free_energy", atoms.calc.results.get("energy"))
    F = atoms.calc.results["forces"]

    if E >= Emax:  # accept or fail
        atoms.positions[...] = atoms.prev_positions

        return []
    else:
        atoms.info["NS_energy"][...] = E
        atoms.arrays["NS_forces"][...] = F

        return []


def walk_lattice_up_down_single(ns_atoms, Emax, rng, move):
    """Walk single atomic positions by lattice parameter while jumping up and down a layer

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        initial atomic configuration
    Emax: float
        maximum shifted energy
    rng: numpy.Generator
        random number generator
    move: str
        id of move

    Returns
    -------
    [] info on move params and attempts/successes
    """
    atoms = ns_atoms.atoms
    atoms.prev_positions[...] = atoms.positions
    atom_id = rng.choice(np.where(atoms.get_tags() >= ns_atoms.tags[move])[0])
    shift = np.array(
        [
            (rng.integers(0, ns_atoms.lattice[0]) + 1 / 2) / ns_atoms.lattice[0],
            (rng.integers(0, ns_atoms.lattice[1]) + 1 / 2) / ns_atoms.lattice[1],
            rng.choice([-1, 1]) * 1.816 / np.linalg.norm(atoms.cell[2]),
        ]
    )
    tmp = atoms.get_scaled_positions(wrap=False)
    tmp[atom_id] += shift

    if not ns_atoms.limit["z"][0] < tmp[atom_id][2] < ns_atoms.limit["z"][1]:
        atoms.positions[...] = atoms.prev_positions
        return []

    atoms.set_scaled_positions(tmp)

    d = neighbor_list(
        "d",
        atoms,
        cutoff=ns_atoms.min_dist,
        self_interaction=False,
    )

    if len(d) > 0:
        atoms.positions[...] = atoms.prev_positions
        return []

    atoms.calc.calculate(
        atoms, properties=["free_energy", "forces"], system_changes=all_changes
    )
    E = atoms.calc.results.get("free_energy", atoms.calc.results.get("energy"))
    F = atoms.calc.results["forces"]

    if E >= Emax:  # accept or fail
        atoms.positions[...] = atoms.prev_positions

        return []
    else:
        atoms.info["NS_energy"][...] = E
        atoms.arrays["NS_forces"][...] = F

        return []


def walk_random_single(ns_atoms, Emax, rng, move):
    """Walk single atomic positions by random assignement

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        initial atomic configuration
    Emax: float
        maximum shifted energy
    rng: numpy.Generator
        random number generator
    move: str
        id of move

    Returns
    -------
    [] info on move params and attempts/successes
    """
    atoms = ns_atoms.atoms
    atoms.prev_positions[...] = atoms.positions
    atom_id = rng.choice(np.where(atoms.get_tags() >= ns_atoms.tags[move])[0])
    tmp = atoms.get_scaled_positions(wrap=False)

    limits = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    for i, val in enumerate(["x", "y", "z"]):
        if val in ns_atoms.limit.keys():
            limits[i] = ns_atoms.limit[val]

    tmp[atom_id] = rng.uniform(limits.T[0], limits.T[1], size=3)
    atoms.set_scaled_positions(tmp)

    d = neighbor_list(
        "d",
        atoms,
        cutoff=ns_atoms.min_dist,
        self_interaction=False,
    )

    if len(d) > 0:
        atoms.positions[...] = atoms.prev_positions
        return []

    atoms.calc.calculate(
        atoms, properties=["free_energy", "forces"], system_changes=all_changes
    )
    E = atoms.calc.results.get("free_energy", atoms.calc.results.get("energy"))
    F = atoms.calc.results["forces"]

    if E >= Emax:  # accept or fail
        atoms.positions[...] = atoms.prev_positions

        return []
    else:
        atoms.info["NS_energy"][...] = E
        atoms.arrays["NS_forces"][...] = F

        return []


def walk_id_swap(ns_atoms, Emax, rng, move):
    """Random swap of two different atom ids

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        initial atomic configuration
    Emax: float
        maximum shifted energy
    rng: numpy.Generator
        random number generator
    move: str
        id of move

    Returns
    -------
    [] info on move params and attempts/successes
    """
    atoms = ns_atoms.atoms
    Zs = list(atoms.numbers)
    symbols = rng.choice(np.unique(atoms.numbers), 2, replace=False)
    tags = atoms.get_tags()
    tags = np.where(tags >= ns_atoms.tags[move], 1, 0)

    id_0 = rng.choice(np.where(atoms.numbers * tags == symbols[0])[0])
    id_1 = rng.choice(np.where(atoms.numbers * tags == symbols[1])[0])

    atoms.numbers[id_0] = symbols[1]
    atoms.numbers[id_1] = symbols[0]

    d = neighbor_list(
        "d",
        atoms,
        cutoff=ns_atoms.min_dist,
        self_interaction=False,
    )

    if len(d) > 0:
        atoms.positions[...] = atoms.prev_positions
        return []

    atoms.calc.calculate(
        atoms, properties=["free_energy", "forces"], system_changes=all_changes
    )
    E = atoms.calc.results.get("free_energy", atoms.calc.results.get("energy"))
    F = atoms.calc.results["forces"]

    if E >= Emax:  # accept or fail
        atoms.set_atomic_numbers(Zs)

        return []
    else:
        atoms.info["NS_energy"][...] = E
        atoms.arrays["NS_forces"][...] = F

        return []


def _min_aspect_ratio(cell):
    """Calculate minimum aspect ratio of cell

    Parameters
    ----------
    cell: float (3,3) np.ndarray
        array of cell row vectors

    Returns
    -------
    minimum_aspect_ratio: float
    """
    crosses = np.asarray([np.cross(cell[i], cell[(i + 1) % 3]) for i in range(3)])
    cross_norms = np.linalg.norm(crosses, axis=1)
    vol = np.abs(np.sum(cell[0] * np.cross(cell[1], cell[2])))
    return np.min(vol / cross_norms) / (vol**one_third)


def _eval_and_accept_or_signal_revert(atoms, Emax, delta_PV=0.0, delta_muN=0.0):
    """Evaluate energy of configuration and accept it, updating NS_energy, NS_forces,
    and NS_energy_sfhit info/arrays entries, or signal that calling routine must revert move

    parameters
    ----------
    atoms: ase.atoms.Atoms
        atomic configuration
    Emax: float
        maximum shifted energy allowed

    Returns
    -------
    revert: bool, True if move is rejected and must be reverted
    """
    atoms.calc.calculate(
        atoms, properties=["free_energy", "forces"], system_changes=all_changes
    )
    E = atoms.calc.results.get("free_energy", atoms.calc.results.get("energy"))
    E_shift = atoms.info["NS_energy_shift"] + delta_PV - delta_muN
    if E + E_shift < Emax:
        # accept can happen here, because it's always the same action
        atoms.info["NS_energy_shift"][...] = E_shift
        atoms.info["NS_energy"][...] = E
        atoms.arrays["NS_forces"][...] = atoms.calc.results["forces"]
        return False
    else:
        # revert can be different, can only signal since it must be in the calling routine
        return True


def walk_cell(ns_atoms, Emax, rng):
    """Walk atom cell using Monte Carlo moves

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        initial atomic configuration
    Emax: float
        maximum shifted energy
    rng: numpy.Generator
        random number generator

    Returns
    -------
    [("cell_volume_per_atom", int n_attempt, int n_success),
     ("cell_shear_per_rt3_atom", int n_attempt, n_success),
     ("cell_stretch", int n_attempt, n_success)] with n_att and n_acc number of attempted and accepted
     move for each submove type
    """
    atoms = ns_atoms.atoms
    N_atoms = len(atoms)
    step_size_volume = ns_atoms.step_size["cell_volume_per_atom"] * N_atoms
    step_size_shear = ns_atoms.step_size["cell_shear_per_rt3_atom"] * (
        N_atoms ** (1.0 / 3.0)
    )
    step_size_stretch = ns_atoms.step_size["cell_stretch"]
    min_aspect_ratio = ns_atoms.move_params["cell"]["min_aspect_ratio"]
    flat_V_prior = ns_atoms.move_params["cell"]["flat_V_prior"]

    n_att = {"volume": 0, "shear": 0, "stretch": 0}
    n_acc = {"volume": 0, "shear": 0, "stretch": 0}

    submoves = list(ns_atoms.move_params["cell"]["submove_probabilities"].keys())
    probs = list(ns_atoms.move_params["cell"]["submove_probabilities"].values())
    for i_step in range(ns_atoms.walk_traj_len["cell"]):
        atoms.prev_cell[...] = atoms.cell.array
        move = rng.choice(submoves, p=probs)
        delta_V = 0.0
        n_att[move] += 1
        if move == "volume":
            orig_V = atoms.get_volume()
            new_V = orig_V + rng.normal(scale=step_size_volume)
            delta_V = new_V - orig_V
            V_scale = new_V / orig_V
            if V_scale < 0.5:
                # too small, reject
                continue
            elif (
                not flat_V_prior
                and new_V < orig_V
                and rng.uniform() > np.pow(V_scale, N_atoms)
            ):
                # V^N prior
                continue
            new_cell = atoms.cell * (V_scale**one_third)
        elif move == "stretch":
            v_ind = rng.integers(3)
            rv = rng.normal(scale=step_size_stretch)
            F = np.eye(3)
            F[v_ind, v_ind] = np.exp(rv)
            v_next = (v_ind + 1) % 3
            F[v_next, v_next] = np.exp(-rv)
            new_cell = atoms.cell @ F
            if _min_aspect_ratio(new_cell) < min_aspect_ratio:
                # min aspect ratio
                continue
        elif move == "shear":
            new_cell = atoms.cell.array.copy()
            v_ind = rng.integers(3)

            v1 = atoms.prev_cell[(v_ind + 1) % 3]
            v1_norm = np.sqrt(np.sum(v1**2))
            dv = rng.normal(scale=step_size_shear) * v1 / v1_norm
            new_cell[v_ind] += dv

            v2 = atoms.prev_cell[(v_ind + 2) % 3]
            v2_norm = np.sqrt(np.sum(v2**2))
            dv = rng.normal(scale=step_size_shear) * v2 / v2_norm
            new_cell[v_ind] += dv
            if _min_aspect_ratio(new_cell) < min_aspect_ratio:
                # min aspect ratio
                continue

        # save positions if needed, then deform
        atoms.prev_positions[...] = atoms.positions
        atoms.set_cell(new_cell, True)

        if _eval_and_accept_or_signal_revert(
            atoms, Emax, delta_PV=ns_atoms.pressure * delta_V
        ):
            atoms.positions[...] = atoms.prev_positions
            atoms.cell.array[...] = atoms.prev_cell
        else:
            n_acc[move] += 1

    return [
        ("cell_volume_per_atom", n_att["volume"], n_acc["volume"]),
        ("cell_shear_per_rt3_atom", n_att["shear"], n_acc["shear"]),
        ("cell_stretch", n_att["stretch"], n_acc["stretch"]),
    ]


def walk_type(ns_atoms, Emax, rng):
    """Walk atom type using swap or semi-Grand-Canonical Monte Carlo moves

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        initial atomic configuration
    Emax: float
        maximum shifted energy
    rng: numpy.Generator
        random number generator

    Returns
    -------
    [] empty list (nominally cell move param attempt/success)
    """
    atoms = ns_atoms.atoms
    N_atoms = len(atoms)
    sGC = ns_atoms.move_params["type"]["sGC"]
    if sGC:
        Zs = list(ns_atoms._Zs)
        n_Zs = len(Zs)
    else:
        # check that swaps are possible, otherwise give up early
        if sum(atoms.numbers == atoms.numbers[0]) == N_atoms:
            return ns_atoms.walk_traj_len["type"], 0

    n_succeeded = 0
    for i_step in range(ns_atoms.walk_traj_len["type"]):
        i0 = rng.integers(N_atoms)
        if sGC:
            Z0 = atoms.numbers[i0]
            type0 = Zs.index(Z0)
            type0_new = (type0 + 1 + rng.integers(n_Zs - 1)) % n_Zs
            atoms.numbers[i0] = Zs[type0_new]

            delta_muN = ns_atoms.mu[Zs[type0_new]] - ns_atoms.mu[Z0]

            if _eval_and_accept_or_signal_revert(atoms, Emax, delta_muN=delta_muN):
                # revert
                atoms.numbers[i0] = Zs[type0]
        else:  # swap
            # pick an atom of a different type
            Z0 = atoms.numbers[i0]
            i1 = rng.choice(np.where(atoms.numbers != Z0)[0])
            atoms.numbers[i0] = atoms.numbers[i1]
            atoms.numbers[i1] = Z0

            if _eval_and_accept_or_signal_revert(atoms, Emax):
                # revert
                Z0 = atoms.numbers[i0]
                atoms.numbers[i0] = atoms.numbers[i1]
                atoms.numbers[i1] = Z0
            else:
                n_succeeded += 1

    return []
