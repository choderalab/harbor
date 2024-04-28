from openeye import oechem, oedepict
import numpy as np


def plot_aligned_ligands(
    filename,
    query_mols: list[oechem.OEMol],
    refmol: oechem.OEMol = None,
    max_width: int = 4,
    quantum_width=150,
    quantum_height=200,
):

    n_ligands = len(query_mols)
    if not refmol:
        # Use the query mol with the largest number of atoms as the MCSS reference
        refmol_idx = 0
        for i, query_mol in enumerate(query_mols):
            refmol = query_mols[refmol_idx]
            refmol_idx = i if query_mol.NumAtoms() > refmol.NumAtoms() else refmol_idx

        refmol = query_mols.pop(i)
        print(refmol.GetTitle(), len(query_mols))

    # Prepare image
    cols = min(max_width, n_ligands)
    rows = int(np.ceil(n_ligands / max_width))
    print(rows, cols)
    image = oedepict.OEImage(quantum_width * cols, quantum_height * rows)
    grid = oedepict.OEImageGrid(image, rows, cols)
    opts = oedepict.OE2DMolDisplayOptions(
        grid.GetCellWidth(), grid.GetCellHeight(), oedepict.OEScale_AutoScale
    )
    opts.SetTitleLocation(oedepict.OETitleLocation_Bottom)
    opts.SetHydrogenStyle(oedepict.OEHydrogenStyle_Hidden)

    refscale = oedepict.OEGetMoleculeScale(refmol, opts)
    oedepict.OEPrepareDepiction(refmol)
    refdisp = oedepict.OE2DMolDisplay(refmol, opts)
    refcell = grid.GetCell(1, 1)
    oedepict.OERenderMolecule(refcell, refdisp)

    # Prep MCS
    atomexpr = (
        oechem.OEExprOpts_Aromaticity
        | oechem.OEExprOpts_AtomicNumber
        | oechem.OEExprOpts_FormalCharge
    )
    bondexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_BondOrder

    # create maximum common substructure object
    pattern_query = oechem.OEQMol(refmol)
    pattern_query.BuildExpressions(atomexpr, bondexpr)
    mcss = oechem.OEMCSSearch(pattern_query)
    mcss.SetMCSFunc(oechem.OEMCSMaxAtoms())
    for i, fitmol in enumerate(query_mols):
        col = (i + 1) % max_width + 1
        row = int(np.ceil((2 + i) / max_width))
        print(row, col)

        alignres = oedepict.OEPrepareAlignedDepiction(fitmol, mcss)
        fitscale = oedepict.OEGetMoleculeScale(fitmol, opts)
        opts.SetScale(refscale)

        # refdisp = oedepict.OE2DMolDisplay(mcss.GetPattern(), opts)
        fitdisp = oedepict.OE2DMolDisplay(fitmol, opts)

        if alignres.IsValid():
            fitabset = oechem.OEAtomBondSet(
                alignres.GetTargetAtoms(), alignres.GetTargetBonds()
            )
            oedepict.OEAddHighlighting(
                fitdisp,
                oechem.OEBlueTint,
                oedepict.OEHighlightStyle_BallAndStick,
                fitabset,
            )
        else:
            raise RuntimeError

        fitcell = grid.GetCell(row, col)
        oedepict.OERenderMolecule(fitcell, fitdisp)
    oedepict.OEWriteImage(filename, image)


def get_mcs_from_mcs_mol(mcs_mol: oechem.OEMol):
    # Prep MCS
    atomexpr = (
        oechem.OEExprOpts_Aromaticity
        | oechem.OEExprOpts_AtomicNumber
        | oechem.OEExprOpts_FormalCharge
        | oechem.OEExprOpts_RingMember

    )
    bondexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_BondOrder

    # create maximum common substructure object
    pattern_query = oechem.OEQMol(mcs_mol)
    pattern_query.BuildExpressions(atomexpr, bondexpr)
    mcss = oechem.OEMCSSearch(pattern_query)
    mcss.SetMCSFunc(oechem.OEMCSMaxAtoms())
    return mcss

def get_n_rows_and_cols(n_mols):
    return int(np.round(np.sqrt(n_mols))), int(np.ceil(np.sqrt(n_mols)))

def get_row_col(i, rows, cols,zero_indexed=True):
    row = i // cols + (0 if zero_indexed else 1)
    col = i % cols + (0 if zero_indexed else 1)
    return int(row), int(col)


def plot_ligands_with_mcs(
    filename: str,
    mcs_mol: oechem.OEMol,
    mols=list[oechem.OEMol],
    max_width: int = 4,
    quantum_width=150,
    quantum_height=200,
    reference="smallest"
):
    # count n ligands + the mcs_mol
    n_ligands = len(mols) # + 1
    print(f"{n_ligands} molecules to plot")
    print([mol.GetTitle() for mol in mols])

    mol_array = np.array(mols)
    n_atoms = np.array([mol.NumAtoms() for mol in mols])
    print(n_atoms)

    if reference == "smallest":
        order = np.argsort(n_atoms)
        mol_array = mol_array[order]
        n_atoms = n_atoms[order]

        smallest_sort = np.argmin(n_atoms)
        refmol = mol_array[smallest_sort]
        mols = np.delete(mol_array, smallest_sort)
    elif reference == "largest":
        order = np.argsort(-n_atoms)
        mol_array = mol_array[order]
        n_atoms = n_atoms[order]
        
        largest_sort = np.argmax(n_atoms)
        refmol = mol_array[largest_sort]
        mols = np.delete(mol_array, largest_sort)
    elif reference == "mcs_mol":
        order = np.argsort(n_atoms)
        mol_array = mol_array[order]
        n_atoms = n_atoms[order]

        mols = mol_array
        refmol = mcs_mol
        n_ligands += 1
    else:
        raise NotImplementedError
    
    # Prepare image
    rows, cols = get_n_rows_and_cols(n_ligands)
    
    print(f"Generating a figure with {rows} rows and {cols} columns")
    image = oedepict.OEImage(quantum_width * cols, quantum_height * rows)
    grid = oedepict.OEImageGrid(image, rows, cols)
    opts = oedepict.OE2DMolDisplayOptions(
        grid.GetCellWidth(), grid.GetCellHeight(), oedepict.OEScale_AutoScale
    )
    opts.SetTitleLocation(oedepict.OETitleLocation_Bottom)
    opts.SetHydrogenStyle(oedepict.OEHydrogenStyle_Hidden)

    refscale = oedepict.OEGetMoleculeScale(refmol, opts)
    

    oedepict.OEPrepareDepiction(refmol)

    mcss = get_mcs_from_mcs_mol(refmol)
    oedepict.OEPrepareAlignedDepiction(refmol, mcss)
    
    opts.SetScale(refscale)
    refdisp = oedepict.OE2DMolDisplay(refmol, opts)
    refcell = grid.GetCell(1, 1)

    oedepict.OERenderMolecule(refcell, refdisp)

    print([mol.GetTitle() for mol in mols])
    for i, fitmol in enumerate(mols):
        i += 1
        row, col = get_row_col(i, rows, cols, zero_indexed=False)
        print(row, col)

        alignres = oedepict.OEPrepareAlignedDepiction(fitmol, mcss)

        if not alignres.IsValid():
            oedepict.OEPrepareDepiction(fitmol)
        
        fitdisp = oedepict.OE2DMolDisplay(fitmol, opts)
        if alignres.IsValid():
            fitabset = oechem.OEAtomBondSet(
                alignres.GetTargetAtoms(), alignres.GetTargetBonds()
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
