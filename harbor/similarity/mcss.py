import numpy as np
from openeye import oechem
from tqdm import tqdm


def get_n_to_n_mcs(refmols: list[oechem.OEMol], querymols: list[oechem.OEMol] = None):
    """
    Get the number of atoms in the maximum common substructure between each pair of molecules.
    :param mols:
    :return:
    """

    # these are the defaaults for atom and bond expressions but just to be explicit I'm putting them here
    atomexpr = (
        oechem.OEExprOpts_Aromaticity
        | oechem.OEExprOpts_AtomicNumber
        | oechem.OEExprOpts_FormalCharge
    )
    bondexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_BondOrder

    if querymols is None:
        querymols = [oechem.OEMol(mol) for mol in refmols]
    refmols = [oechem.OEMol(mol) for mol in refmols]

    # TODO: halve the compute time by only computing the upper triangle of the matrix

    # Set up the search pattern and MCS objects
    mcs_num_atoms = np.zeros((len(refmols), len(querymols)), dtype=int)
    for i, refmol in tqdm(enumerate(refmols), total=len(refmols)):
        pattern_query = oechem.OEQMol(refmol)
        pattern_query.BuildExpressions(atomexpr, bondexpr)
        mcss = oechem.OEMCSSearch(pattern_query)
        mcss.SetMCSFunc(oechem.OEMCSMaxAtomsCompleteCycles())

        for j, querymol in enumerate(querymols):
            # MCS search
            try:
                mcs = next(iter(mcss.Match(querymol, True)))
                mcs_num_atoms[i, j] = mcs.NumAtoms()
            except StopIteration:  # no match found
                mcs_num_atoms[i, j] = 0
    return mcs_num_atoms


def get_mcs_substructure(refmol: oechem.OEMol, querymol: oechem.OEMol):
    # these are the defaaults for atom and bond expressions but just to be explicit I'm putting them here
    atomexpr = (
        oechem.OEExprOpts_Aromaticity
        | oechem.OEExprOpts_AtomicNumber
        | oechem.OEExprOpts_FormalCharge
    )
    bondexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_BondOrder

    # Set up the search pattern and MCS objects
    pattern_query = oechem.OEQMol(refmol)
    pattern_query.BuildExpressions(atomexpr, bondexpr)
    mcss = oechem.OEMCSSearch(pattern_query)
    mcss.SetMCSFunc(oechem.OEMCSMaxAtomsCompleteCycles())
    mcs = next(iter(mcss.Match(querymol, True)))
    core_fragment = oechem.OEMol()
    oechem.OESubsetMol(core_fragment, mcs)
    return core_fragment


def get_mcs_mol(mol1: oechem.OEMol, mol2: oechem.OEMol):
    # Set atom and bond expressions
    atomexpr = (
        oechem.OEExprOpts_Aromaticity
        | oechem.OEExprOpts_AtomicNumber
        | oechem.OEExprOpts_FormalCharge
    )
    bondexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_BondOrder

    # Set up the search pattern and MCS objects
    pattern_query = oechem.OEQMol(mol1)
    pattern_query.BuildExpressions(atomexpr, bondexpr)
    mcss = oechem.OEMCSSearch(pattern_query)
    mcss.SetMCSFunc(oechem.OEMCSMaxAtomsCompleteCycles())
    try:
        new_mcs = next(iter(mcss.Match(mol2, True)))
        core_fragment = oechem.OEMol()
        if not oechem.OESubsetMol(core_fragment, mol2, new_mcs):
            raise RuntimeError(
                f"Failed to generate mcs for {mol1.GetTitle()} & {mol2.GetTitle()}"
            )
    except StopIteration:
        raise RuntimeError
    return core_fragment


def get_mcs_from_mcs_mol(mcs_mol: oechem.OEMol):
    # Prep MCS
    atomexpr = (
        oechem.OEExprOpts_Aromaticity
        | oechem.OEExprOpts_AtomicNumber
        | oechem.OEExprOpts_FormalCharge
    )
    bondexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_BondOrder

    # create maximum common substructure object
    pattern_query = oechem.OEQMol(mcs_mol)
    pattern_query.BuildExpressions(atomexpr, bondexpr)
    mcss = oechem.OEMCSSearch(pattern_query)
    mcss.SetMCSFunc(oechem.OEMCSMaxAtoms())
    return mcss
