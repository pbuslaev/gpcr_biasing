import glob
import os

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import distances as dist_analysis

PAIRS = "/home/pbuslaev/projects/personal/gpcr/residue_pairs.dat"


# flake8: noqa: max-complexity
def create_resid_generic_mapping(pdb_file1, pdb_file2):
    """
    Create a dictionary mapping resid to generic numbers based on two PDB
    files.

    Parameters
    ----------
    pdb_file1 : str
        Path to the first PDB file (structure file)
    pdb_file2 : str
        Path to the second PDB file (file with generic numbers in b-factor)

    Returns
    -------
    dict
        Dictionary with resid as key and generic number (nxnn format) as value

    Raises
    ------
    ValueError
        If CA atom counts differ or residue mismatches are found
    """
    # Load both PDB files
    u1 = mda.Universe(pdb_file1)
    u2 = mda.Universe(pdb_file2)

    # Select CA atoms
    ca1 = u1.select_atoms("name CA")
    ca2 = u2.select_atoms("name CA")

    # Check that number of CA atoms is the same
    if len(ca1) != len(ca2):
        raise ValueError(
            f"Number of CA atoms differs: {len(ca1)} vs {len(ca2)}"
        )

    # Create mapping dictionary
    resid_generic_map = {}

    for atom1, atom2 in zip(ca1, ca2):
        # Check that chain, resname, and resid match
        if (
            atom1.segid != atom2.segid
            or atom1.resname != atom2.resname
            or atom1.resid != atom2.resid
        ):
            raise ValueError(
                "Residue mismatch: "
                f"({atom1.segid}, {atom1.resname}, {atom1.resid}) vs "
                f"({atom2.segid}, {atom2.resname}, {atom2.resid})"
            )

        # Convert generic number from n.nn format to nxnn format
        if atom2.tempfactor > 0:  # Only process if there's a generic number
            generic_str = f"{atom2.tempfactor:.2f}".replace(".", "x")
            resid_generic_map[int(atom1.resid)] = generic_str

    return resid_generic_map


def calculate_distances_from_pairs(
    topology, trajectory, mapping, pairs_file, idx
):
    """
    Calculate distances between CA/CB atoms for residue pairs from a PAIRS file.

    Parameters
    ----------
    topology : str
        Path to the topology file
    trajectory : str
        Path to the trajectory file
    mapping : dict
        Dictionary mapping resid to generic numbers
    pairs_file : str
        Path to file containing pairs of generic numbers (one pair per line)
    idx : int
        Id of the trajectory used.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns for each pair and rows for each frame
    """

    # Load universe
    u = mda.Universe(topology, trajectory)

    # Read pairs from file
    pairs = []
    with open(pairs_file, "r") as f:
        for line in f:
            line = line.strip()
            parts = line.split(":")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))

    # Create reverse mapping (generic -> resid)
    reverse_mapping = {v: k for k, v in mapping.items()}

    # Iterate through trajectory
    # Pre-select atom groups for each pair
    ag1 = None
    ag2 = None
    pair_labels = []

    for gen1, gen2 in pairs:
        if gen1 not in reverse_mapping:
            raise ValueError(f"Key not on mapping: {gen1}")
        if gen2 not in reverse_mapping:
            raise ValueError(f"Key not on mapping: {gen2}")

        resid1 = reverse_mapping[gen1]
        resid2 = reverse_mapping[gen2]

        # Try to select CB, fallback to CA
        for name1 in ["CA", "CB"]:
            atom1 = u.select_atoms(f"resid {resid1} and name {name1}")
            if len(atom1) == 0:
                continue
            for name2 in ["CA", "CB"]:
                atom2 = u.select_atoms(f"resid {resid2} and name {name2}")
                if len(atom2) == 0:
                    continue
                pair_labels.append(f"{gen1}:{name1}-{gen2}:{name2}")
                ag1 = atom1 if ag1 is None else ag1 + atom1
                ag2 = atom2 if ag2 is None else ag2 + atom2

    # Calculate distances for all frames

    dists = []
    for ts in u.trajectory:
        d = dist_analysis.dist(ag1, ag2)[2]
        dists.append(d)

    dists = np.array(dists)
    dists = pd.DataFrame(dists, columns=pair_labels)
    dists["idx"] = idx

    return dists


### Testing
if __name__ == "__main__":
    trajs = [
        "id_123",
        "id_83",
        "id_99",
        "id_115",
        "id_117",
        "id_121",
        "id_124",
        "id_160",
    ]
    distances = None
    for f in trajs:
        id = int(f.split("_")[1])
        print(f"Mapping for trajectory {id}")
        pdb1 = glob.glob(os.path.join(f, f"*{id}.pdb"))[0]
        psf = glob.glob(os.path.join(f, f"*{id}.psf"))[0]
        pdb2 = glob.glob(os.path.join(f, f"*{id}_GPCRDB.pdb"))[0]
        trj = glob.glob(os.path.join(f, f"*{id}.xtc"))[0]

        mapping = create_resid_generic_mapping(pdb1, pdb2)
        backmaping = {v: k for k, v in mapping.items()}
        # print(mapping)
        # print(backmaping)
        idx = int(f.split("_")[1])
        tmp_distances = calculate_distances_from_pairs(
            psf, trj, mapping, PAIRS, idx
        )
        if distances is None:
            distances = tmp_distances
        else:
            distances = pd.concat(
                [distances, tmp_distances], ignore_index=True
            )
    distances.to_csv("dataset.csv", index=False)
