def get_fp(mol, bit_size=2048, radius=2):
    from openeye import oegraphsim

    fp = oegraphsim.OEFingerPrint()
    oegraphsim.OEMakeCircularFP(
        fp,
        mol,
        bit_size,
        0,
        radius,
        oegraphsim.OEFPAtomType_DefaultCircularAtom,
        oegraphsim.OEFPBondType_DefaultCircularBond,
    )
    return fp
