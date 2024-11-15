from pathlib import Path
from argparse import ArgumentParser
import random
from rdkit import Chem
from openeye import oechem
from harbor.clustering import hierarchical as h


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
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=50,
        description="Maximum number of iterations for clustering",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=15,
        description="Number of atoms to use to cluster molecules",
    )
    return parser.parse_args()


def main():
    args = get_args()

    mols = Chem.SDMolSupplier(str(args.input_sdf))

    oemols = []
    mol_ids = []
    for rdkit_mol in random.sample(mols, 15):
        smiles = Chem.MolToSmiles(rdkit_mol)
        properties = rdkit_mol.GetPropsAsDict()
        mol_ids.append(properties["compound_name"])
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smiles)
        oemols.append(mol)

    clusterer = h.HeirarchicalClustering(molecules=oemols, mol_ids=mol_ids)

    clusters = clusterer.cluster(max_iterations=50, cutoff=15)


if __name__ == "__main__":
    main()
