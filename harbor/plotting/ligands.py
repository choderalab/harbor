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
        | oechem.OEExprOpts_RingMember
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
