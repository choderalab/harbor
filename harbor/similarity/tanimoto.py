from openeye import oegraphsim


def calculate_tanimoto(fp1: oegraphsim.OETanimoto, fp2: oegraphsim.OETanimoto):
    """
    Calculate the Tanimoto similarity between two fingerprints. Calculate fingerprints using the get_fp function.
    :param fp1:
    :param fp2:
    :return:
    """
    return oegraphsim.OETanimoto(fp1, fp2)
