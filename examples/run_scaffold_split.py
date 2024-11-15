from argparse import ArgumentParser
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from openeye import oechem, oedepict
from collections import defaultdict
import random
import numpy as np
import tqdm


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_sdf",
        type=Path,
        help="Path to input SDF file containing ligands",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        help="Path to output directory",
    )
    return parser.parse_args()


def generate_scaffold(mol, include_chirality=True):
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.
    Implementation copied from https://github.com/chemprop/chemprop.

    :param mol: A smiles string or an RDKit molecule.
    :param include_chirality: Whether to include chirality.
    :return:
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol
    )  # , includeChirality=include_chirality

    return scaffold


def scaffold_to_smiles(mols, use_indices):
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.
    Implementation copied from https://github.com/chemprop/chemprop.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in enumerate(mols):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def get_murcko_scaffold_match(scaffold: oechem.OEMol, fitmol: oechem.OEMol):
    # Get Match
    scaffold_smiles = oechem.OEMolToSmiles(scaffold)
    ss = oechem.OESubSearch(scaffold_smiles)
    oechem.OEPrepareSearch(scaffold, ss)
    if ss.SingleMatch(fitmol):
        pattern_match = list(ss.Match(fitmol))[0]
        found = True
    else:
        found = False
        pattern_match = None
    return found, pattern_match


def get_murcko_scaffold_match_rdkit(scaffold: Chem.Mol, fitmol: Chem.Mol):
    atom_indices = fitmol.GetSubstructureMatch(scaffold)


def convert_to_oechem(scaffolds) -> list[dict]:
    return_list = []
    for scaffold, mols in tqdm.tqdm(scaffolds.items()):
        scaffold_oemol = oechem.OEMol()
        oechem.OESmilesToMol(scaffold_oemol, scaffold)
        mol_list = []
        for mol in mols:
            smiles = Chem.MolToSmiles(mol)
            oemol = oechem.OEMol()
            oechem.OESmilesToMol(oemol, smiles)
            mol_list.append(oemol.__copy__())
        return_list.append({"scaffold": scaffold_oemol.__copy__(), "mols": mol_list})
    return return_list


def plot_murcko_scaffold_aligned_ligands(
    filename,
    query_mols: list[oechem.OEMol],
    scaffold: oechem.OEMol,
    max_width: int = 4,
    quantum_width=150,
    quantum_height=200,
):
    n_ligands = len(query_mols) + 1

    # Prepare image
    cols = min(max_width, n_ligands)
    rows = int(np.ceil(n_ligands / max_width))
    image = oedepict.OEImage(quantum_width * cols, quantum_height * rows)
    grid = oedepict.OEImageGrid(image, rows, cols)
    opts = oedepict.OE2DMolDisplayOptions(
        grid.GetCellWidth(), grid.GetCellHeight(), oedepict.OEScale_AutoScale
    )
    opts.SetTitleLocation(oedepict.OETitleLocation_Bottom)
    opts.SetHydrogenStyle(oedepict.OEHydrogenStyle_Hidden)

    oedepict.OEPrepareDepiction(scaffold)

    # Get Scale
    minscale = float("inf")
    for mol in query_mols:
        oedepict.OEPrepareDepiction(mol)
        minscale = min(minscale, oedepict.OEGetMoleculeScale(mol, opts))
    opts.SetScale(minscale)

    print(f"minscale: {minscale}")

    # Draw Reference
    refdisp = oedepict.OE2DMolDisplay(scaffold, opts)
    refcell = grid.GetCell(1, 1)
    oedepict.OERenderMolecule(refcell, refdisp)

    oechem.OESuppressHydrogens(scaffold)

    for i, fitmol in enumerate(query_mols):

        # Get Match
        oechem.OESuppressHydrogens(fitmol)
        scaffold_smiles = oechem.OEMolToSmiles(scaffold)
        ss = oechem.OESubSearch(scaffold_smiles)
        oechem.OEPrepareSearch(scaffold, ss)
        if ss.SingleMatch(fitmol):
            pattern_match = list(ss.Match(fitmol))[0]
            found = True
        else:
            found = False
            pattern_match = None

        col = (i + 1) % max_width + 1
        row = int(np.ceil((2 + i) / max_width))
        print(f"{rows}: {row}; {cols}: {col}")

        oedepict.OEPrepareDepiction(fitmol)

        if found:
            align_result = oedepict.OEPrepareAlignedDepiction(
                fitmol, scaffold, pattern_match
            )
        else:
            align_result = oedepict.OEPrepareDepiction(fitmol)

        fitdisp = oedepict.OE2DMolDisplay(fitmol, opts)

        if found:
            fitabset = oechem.OEAtomBondSet(
                pattern_match.GetTargetAtoms(), pattern_match.GetTargetBonds()
            )
            oedepict.OEAddHighlighting(
                fitdisp,
                oechem.OEBlueTint,
                oedepict.OEHighlightStyle_BallAndStick,
                fitabset,
            )

        fitcell = grid.GetCell(row, col)
        oedepict.OERenderMolecule(fitcell, fitdisp)
    oedepict.OEWriteImage(filename, image)


def main():
    args = get_args()

    mols = Chem.SDMolSupplier(str(args.input_sdf))
    scaffolds = scaffold_to_smiles(mols, use_indices=False)
    oechem_list = convert_to_oechem(dict(scaffolds))

    # sort list by number of mols
    oechem_list = sorted(oechem_list, key=lambda x: len(x["mols"]), reverse=True)

    for i, scaffold in enumerate(oechem_list):
        plot_murcko_scaffold_aligned_ligands(
            filename=f"scaffold_{i}.png",
            query_mols=scaffold["mols"],
            scaffold=scaffold["scaffold"],
            quantum_width=400,
            quantum_height=400,
        )


if __name__ == "__main__":
    main()
