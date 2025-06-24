from pydantic import BaseModel, Field, root_validator
from enum import Enum, auto
from plip.structure.preparation import PDBComplex, PLInteraction
import pandas as pd
from pathlib import Path
import json

class InteractionType(Enum):
    """
    Protein-Ligand Interaction Types. Names refer to the role of the Ligand
    """
    HydrogenBondDonor = 'HydrogenBondDonor'
    HydrogenBondAcceptor = 'HydrogenBondAcceptor'
    HydrophobicInteraction = 'HydrophobicInteraction'
    PiStacking = 'PiStacking'
    HalogenBond = 'HalogenBond'
    SaltBridge = 'SaltBridge'

    def __str__(self):
        return self.value

class AtomType(str, Enum):
    NONE = 'none'
    H = 'H'
    C = 'C'
    N = 'N'
    O = 'O'
    S = 'S'
    P = 'P'
    F = 'F'
    Cl = 'Cl'
    Br = 'Br'

    def __str__(self):
        return self.value

class ProteinResidueType(str, Enum):
    ALA = 'ALA'
    ARG = 'ARG'
    ASN = 'ASN'
    ASP = 'ASP'
    CYS = 'CYS'
    GLN = 'GLN'
    GLU = 'GLU'
    GLY = 'GLY'
    HIS = 'HIS'
    ILE = 'ILE'
    LEU = 'LEU'
    LYS = 'LYS'
    MET = 'MET'
    PHE = 'PHE'
    PRO = 'PRO'
    SER = 'SER'
    THR = 'THR'
    TRP = 'TRP'
    TYR = 'TYR'
    VAL = 'VAL'

    def __str__(self):
        return self.value

class FormalCharge(str, Enum):
    Positive = 'positive'
    Negative = 'negative'
    Neutral = 'neutral'

    def __str__(self):
        return self.name


class ProteinLigandInteraction(BaseModel):
    interaction_type: InteractionType = Field(..., description="Type of interaction between protein and ligand.")
    protein_residue_number: int = Field(..., description="Residue number of the protein involved in the interaction.")
    protein_residue_type: ProteinResidueType = Field(..., description="Amino acid identify of the residue involved in the interaction.")
    to_sidechain: bool = Field(..., description="Indicates if the interaction is to the sidechain of the protein residue (if False, it is to the backbone).")
    count: int = Field(1, description="Count of the number of interactions of this type.")
    ligand_atom_type: AtomType = Field('none', description="Type of the ligand atom involved in the interaction.")
    protein_atom_type: AtomType = Field('none', description="Type of the protein atom involved in the interaction.")
    ligand_charge: FormalCharge = Field('neutral', description="Formal charge of the ligand involved in the interaction.")
    protein_charge: FormalCharge = Field('neutral', description="Formal charge of the protein involved in the interaction.")
    protein_atom_number: int = Field(-1,
                                     description="Atom number of the protein involved in the interaction. Optional, can be -1 if not applicable.")
    ligand_atom_number: int = Field(-1, description= "Atom number of the ligand involved in the interaction. Optional, can be -1 if not applicable.")

    def to_json_dict(self) -> dict:
        return {k: v.value if k in [self.interaction_type, self.protein_residue_type] else v for k, v in self.dict().items()}


def plip_constructor(plip: PLInteraction) -> list[ProteinLigandInteraction]:
    """
    Function to get a tidy dataframe of interactions from a plip fingerprint

    Parameters
    ----------
    plip : PLInteraction
        The PLIP interaction object containing all interaction types for a given binding site.

    Returns
    -------
    list[ProteinLigandInteraction]
    """

    interactions = []
    for interaction in plip.all_itypes:
        interaction_type = type(interaction).__name__

        protein_atom_type = AtomType.NONE
        ligand_atom_type = AtomType.NONE
        ligand_charge = FormalCharge.Neutral
        protein_charge = FormalCharge.Neutral
        protein_atom_number = -1
        ligand_atom_number = -1



        if interaction_type == 'hbond':
            output_interaction_type = InteractionType.HydrogenBondAcceptor if interaction.protisdon else InteractionType.HydrogenBondDonor
            protein_atom_type = AtomType(interaction.dtype[0] if interaction.protisdon else interaction.atype[0])
            ligand_atom_type = AtomType(interaction.atype[0] if interaction.protisdon else interaction.dtype[0])
            protein_atom_number = interaction.a_orig_idx if interaction.protisdon else interaction.d_orig_idx
            ligand_atom_number = interaction.d_orig_idx if interaction.protisdon else interaction.a_orig_idx

        elif interaction_type == 'halogenbond':
            # We assume that halogen bonds are always ligand-donor protein-acceptor
            output_interaction_type = InteractionType.HalogenBond
            protein_atom_type = AtomType(interaction.acctype[0])
            ligand_atom_type = AtomType(interaction.donortype[0])
            protein_atom_number = interaction.acc_orig_idx
            ligand_atom_number = interaction.don_orig_idx

        elif interaction_type == 'saltbridge':
            output_interaction_type = InteractionType.SaltBridge
            ligand_charge = FormalCharge.Negative if interaction.protispos else FormalCharge.Positive
            protein_charge = FormalCharge.Positive if interaction.protispos else FormalCharge.Negative

        elif interaction_type == 'hydroph_interaction':
            output_interaction_type = InteractionType.HydrophobicInteraction

        elif interaction_type == 'pication':
            output_interaction_type = InteractionType.PiStacking
            protein_charge = FormalCharge.Positive if interaction.protcharged else FormalCharge.Negative

        elif interaction_type == 'pistack':
            output_interaction_type = InteractionType.PiStacking

        else:
            raise NotImplementedError(f"Interaction type not implemented: {interaction_type} for interaction: {interaction}")

        protein_residue_number = interaction.resnr
        protein_residue_type = interaction.restype

        # These interactions must be to the sidechain, and therefore PLIP does not provide the sidechain boolean
        if interaction_type in ['hydroph_interaction', 'pistack', 'saltbridge', 'pication']:
            to_sidechain = True
        else:
            to_sidechain = interaction.sidechain

        interactions.append(ProteinLigandInteraction(interaction_type=output_interaction_type,
                                                     protein_residue_number=protein_residue_number,
                                                     protein_residue_type=protein_residue_type,
                                                     to_sidechain=to_sidechain,
                                                     ligand_atom_type=ligand_atom_type,
                                                     protein_atom_type=protein_atom_type,
                                                     ligand_charge=ligand_charge,
                                                     protein_charge=protein_charge,
                                                    protein_atom_number=protein_atom_number,
                                                    ligand_atom_number=ligand_atom_number,
                                                     ))
    return interactions


def collect_duplicates(plints: list[ProteinLigandInteraction]) -> list[ProteinLigandInteraction]:
    """
    This function takes a list of ProteinLigandInteractions and
    returns a list of unique interactions with the count updated if any duplicates are found.

    Parameters
    ----------
    plints : list[ProteinLigandInteraction]
        A list of ProteinLigandInteraction objects, potentially with duplicates.

    Returns
    -------
    list[ProteinLigandInteraction]
    """
    obj_dict = {}

    for obj in plints:
        if obj.json() in obj_dict:
            obj_dict[obj.json()].count += 1
        else:
            obj_dict[obj.json()] = obj

    return list(obj_dict.values())


class PLIntReport(BaseModel):
    """
    Class to store a report of PLIP interactions

    Parameters
    ----------
    structure : str
        The path to the PDB structure file.
    interactions : list[ProteinLigandInteraction]
        A list of ProteinLigandInteraction objects representing the interactions found in the structure.

    """
    report_id: str = Field(..., description="Identifier for the structure, typically the input structure filename without extension.")
    structure: str = Field(..., description="Path to the PDB structure file.")
    interactions: list[ProteinLigandInteraction] = Field( ..., description="List of Protein-Ligand Interaction objects found in the structure.")

    @classmethod
    def from_complex_path(cls, complex_path: str | Path, ligand_id="UNK") -> "PLIntReport":
        """
        Create a PLIntReport from a PDBComplex object loaded from a file path.

        Parameters
        ----------
        complex_path : str | Path
            The path to the PDB file containing the protein-ligand complex.
        ligand_id : str, optional
            The identifier of the ligand in the complex (default is "UNK").

        Returns
        -------
        PLIntReport
            An instance of PLIntReport containing the structure path and a list of ProteinLigandInteraction objects.
        """
        my_mol = PDBComplex()
        my_mol.load_pdb(str(complex_path))
        my_mol.analyze()

        binding_site = None

        for k, v in my_mol.interaction_sets.items():
            if ligand_id in k:
                binding_site = k
                break

        # Convert the PLIP interaction analysis of the binding site to a list of ProteinLigandInteractions and then
        # Collect any duplicate interactions
        if binding_site:
            raw_plip_report = my_mol.interaction_sets[binding_site]
            interactions = collect_duplicates(plip_constructor(raw_plip_report))
        else:
            interactions = []

        return cls(report_id=complex_path.stem, structure=str(complex_path), interactions=interactions)

    def to_csv(self, path: str | Path):
        """
        Save the interactions to a CSV file.

        Parameters
        ----------
        path : str | Path
            The path where the CSV file will be saved.
        """
        df = pd.DataFrame.from_records([json.loads(interaction.json()) for interaction in self.interactions])
        df['report_id'] = self.report_id  # Add report_id to each row for reference
        df['structure'] = self.structure  # Add structure path to each row for reference
        df.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, plint_report: str | Path) -> "PLIntReport":
        """
        Create a PLIntReport from a CSV file.

        Parameters
        ----------
        plint_report : str | Path
            The path to the CSV file containing the interactions.

        Returns
        -------
        PLIntReport
            An instance of PLIntReport containing the structure path and a list of ProteinLigandInteraction objects.
        """
        df = pd.read_csv(plint_report)
        report_id = df['report_id'].iloc[0] if 'report_id' in df.columns else Path(plint_report).stem
        pdb_file = df['structure'].iloc[0] if 'structure' in df.columns else str(plint_report)

        interactions = [ProteinLigandInteraction(**row) for _, row in df.iterrows()]
        return PLIntReport(report_id=report_id, structure=pdb_file, interactions=interactions)


class FingerprintLevel(Enum):
    """
    Enum to specify the level of detail for the fingerprint
    """
    ByTotalInteractions = 'ByTotalInteractions'
    ByInteractionType = 'ByInteractionType'
    ByInteractionTypeAndResidueType = 'ByInteractionTypeAndResidueType'
    ByInteractionTypeAndAtomTypes = 'ByInteractionTypeAndAtomTypes'
    ByInteractionTypeAndResidueTypeAndBBorSC = 'ByInteractionTypeAndResidueTypeAndBBorSC'
    ByInteractionTypeAndResidueTypeAndNumber = 'ByInteractionTypeAndResidueTypeAndNumber'
    ByEverything = 'ByEverything'

    def __str__(self):
        return self.value


def calculate_fingerprint(plint_report: PLIntReport, level: FingerprintLevel) -> dict:
    """
    Calculate a fingerprint of the protein-ligand interactions based on the specified level of detail.

    Parameters
    ----------
    plint_report : PLIntReport
        A report containing a list of protein-ligand interactions.
    level : FingerprintLevel
        The level of detail for the fingerprint. Options include:
        - ByTotalInteractions: Returns total count of interactions
        - ByInteractionType: Groups by type (hydrogen bond, salt bridge, etc.)
        - ByInteractionTypeAndResidueType: Groups by type and residue (e.g., HBond_ALA)
        - ByInteractionTypeAndAtomTypes: Groups by type and atom types involved
        - ByInteractionTypeAndResidueTypeAndBBorSC: Groups by type, residue and backbone/sidechain
        - ByInteractionTypeAndResidueTypeAndNumber: Groups by type, residue and residue number
        - ByEverything: Creates unique key for each unique interaction property combination

    Returns
    -------
    dict
        A dictionary where keys are interaction descriptors based on the specified level
        and values are the counts of those interactions. For example:
        - ByTotalInteractions: {'TotalInteractions': 5}
        - ByInteractionType: {'HydrogenBond': 2, 'SaltBridge': 3}
        - ByInteractionTypeAndResidueType: {'HydrogenBond_ALA': 1, 'HydrogenBond_SER': 1}

    Raises
    ------
    ValueError
        If an invalid FingerprintLevel is provided.
    """
    fingerprint_dict = {}
    if level == FingerprintLevel.ByTotalInteractions:
        fingerprint_dict['TotalInteractions'] = len(plint_report.interactions)
    else:
        for interaction in plint_report.interactions:
            if level == FingerprintLevel.ByInteractionType:
                key = interaction.interaction_type.value
            elif level == FingerprintLevel.ByInteractionTypeAndResidueType:
                key = f"{interaction.interaction_type.value}_{interaction.protein_residue_type}"
            elif level == FingerprintLevel.ByInteractionTypeAndAtomTypes:
                key = f"{interaction.interaction_type.value}_Protein_{interaction.protein_atom_type}_Ligand_{interaction.ligand_atom_type}"
            elif level == FingerprintLevel.ByInteractionTypeAndResidueTypeAndBBorSC:
                key = f"{interaction.interaction_type.value}_{interaction.protein_residue_type}_{'SC' if interaction.to_sidechain else 'BB'}"
            elif level == FingerprintLevel.ByInteractionTypeAndResidueTypeAndNumber:
                key = f"{interaction.interaction_type.value}_{interaction.protein_residue_type}{interaction.protein_residue_number}"
            elif level == FingerprintLevel.ByEverything:
                key = "_".join([f"{k}_{str(v)}" for k, v in interaction.dict().items()])
            else:
                raise ValueError("Invalid Fingerprint Level")
            original_count = fingerprint_dict.get(key, 0)
            fingerprint_dict[key] = original_count + interaction.count
    return fingerprint_dict


class SimilarityScore(BaseModel):
    """
    Class to store a similarity score between two fingerprints with some record of how it was calculated.
    """
    provenance: str = Field(..., description="Method used to calculate the similarity score.")
    score: float = Field(..., description="Similarity score between the two fingerprints, ranging from 0 to 1.")
    number_of_interactions_in_reference: int = Field(..., description="Total number of interactions in the reference fingerprint.")
    number_of_interactions_in_query: int = Field(..., description="Total number of interactions in the query fingerprint.")
    number_of_interactions_in_intersection: int = Field(..., description="Number of interactions common to both fingerprints.")
    number_of_interactions_in_union: int = Field(..., description="Total number of unique interactions across both fingerprints.")

    @root_validator()
    def check_score_range(cls, values):
        """
        Validate that the score is between 0 and 1.
        """
        score = values.get('score')
        if not (0 <= score <= 1):
            raise ValueError("Score must be between 0 and 1.")
        return values

    @root_validator()
    def check_interaction_counts(cls, values):
        """
        Validate that the interaction counts are non-negative and make sense.
        """
        num_ref = values.get('number_of_interactions_in_reference')
        num_query = values.get('number_of_interactions_in_query')
        num_intersection = values.get('number_of_interactions_in_intersection')
        num_union = values.get('number_of_interactions_in_union')

        if num_ref < 0 or num_query < 0 or num_intersection < 0 or num_union < 0:
            raise ValueError("Interaction counts must be non-negative.")

        if not (num_intersection <= num_ref and num_intersection <= num_query):
            raise ValueError(f"Number of interactions in intersection ({num_intersection}) cannot exceed those in reference ({num_ref}) or query ({num_query}).")

        if not (num_union >= max(num_ref, num_query)):
            raise ValueError(f"Number of interactions in union ({num_union}) must be at least as large as the larger of the two fingerprints (ref {num_ref}, query {num_query}).")

        if not num_ref + num_query - num_intersection == num_union:
            raise ValueError(f"Union {num_union} must equal the sum of both fingerprints (ref {num_ref}, query {num_query}) minus the intersection ({num_intersection}).")

        return values


def calculate_tversky(fingerprint1: dict, fingerprint2: dict, alpha: float = 1, beta: float = 0) -> SimilarityScore:
    """
    Calculate the Tversky Index between two fingerprints.

    The Tversky Index is an asymmetric similarity measure that generalizes several other similarity coefficients:
    Tversky Index = |A ∩ B| / (|A ∩ B| + α|A - B| + β|B - A|)
    - For Recall: α=1, β=0
    - For Tanimoto Coefficient: α=β=1
    - For Dice Coefficient: α=β=0.5

    Parameters
    ----------
    fingerprint1 : dict
        Reference fingerprint dictionary with features as keys and counts as values.
    fingerprint2 : dict
        Query fingerprint dictionary with features as keys and counts as values.
    alpha : float, optional
        Weight for features unique to fingerprint1 (default=1).
    beta : float, optional
        Weight for features unique to fingerprint2 (default=0).

    Returns
    -------
    SimilarityScore
        Object containing the similarity score and additional metrics about the comparison.
    """
    # First get the union of the keys
    fp_types = set(fingerprint1.keys()).union(set(fingerprint2.keys()))

    # Calculate the number of interactions that are in both fingerprints
    matched = sum([min(fingerprint1.get(a, 0), fingerprint2.get(a, 0)) for a in fp_types])

    # Calculate the number of interactions in the union
    union = sum(fingerprint1.values()) + sum(fingerprint2.values()) - matched
    union_v2 = sum([max(fingerprint1.get(a, 0), fingerprint2.get(a, 0)) for a in fp_types])
    if not union == union_v2:
        raise ValueError(f"Union calculation mismatch: {union} != {union_v2}. Please check the input fingerprints.")

    # Calculate the Tversky Index
    score = matched / (matched + alpha * (sum(fingerprint1.values()) - matched) + beta * (
                sum(fingerprint2.values()) - matched))
    return SimilarityScore(provenance=f"Tversky_alpha{alpha:.2f}_beta{beta:.2f}", score=score,
                           number_of_interactions_in_reference=sum(fingerprint1.values()),
                           number_of_interactions_in_query=sum(fingerprint2.values()),
                           number_of_interactions_in_intersection=matched, number_of_interactions_in_union=union, )


class InteractionScore(BaseModel):
    """
    Class to store the results of comparing two PLIntReports.
    Automatically calculates the Tanimoto Coefficient (alpha=beta=1) and the Tversky Index (alpha=1, beta=0)
    """
    provenance: str = Field(..., description="Level of detail used for the fingerprint comparison (e.g., ByInteractionType).")
    number_of_interactions_in_query: int = Field(..., description="Total number of interactions in the query structure.")
    number_of_interactions_in_reference: int = Field(..., description="Total number of interactions in the reference structure.")
    number_of_interactions_in_intersection: int = Field(..., description="Number of interactions common to both structures.")
    number_of_interactions_in_union: int = Field(..., description="Total number of unique interactions across both structures.")
    tanimoto_coefficient: float = Field(..., description="Tanimoto coefficient (α=β=1) measuring similarity between the structures, ranges from 0 to 1.")
    tversky_index: float = Field(..., description="Tversky index (α=1, β=0) measuring how well the query matches the reference, ranges from 0 to 1.")
    reference_fingerprint: dict = Field(..., description="Dictionary containing the interaction fingerprint of the reference structure.")
    query_fingerprint: dict = Field(..., description="Dictionary containing the interaction fingerprint of the query structure.")

    @classmethod
    def from_fingerprints(cls, reference: PLIntReport, query: PLIntReport,
                          level: FingerprintLevel) -> "InteractionScore":

        # Calculate the fingerprints according to the specified level of detail
        reference = calculate_fingerprint(reference, level)
        query = calculate_fingerprint(query, level)

        # Calculate the Tanimoto Coefficient and Tversky Index
        tanimoto = calculate_tversky(reference, query, alpha=1, beta=1)
        tversky_index = calculate_tversky(reference, query)

        return cls(provenance=level.value, number_of_interactions_in_query=tanimoto.number_of_interactions_in_query,
                   number_of_interactions_in_reference=tanimoto.number_of_interactions_in_reference,
                   number_of_interactions_in_intersection=tanimoto.number_of_interactions_in_intersection,
                   number_of_interactions_in_union=tanimoto.number_of_interactions_in_union,
                   tanimoto_coefficient=tanimoto.score,
                   tversky_index=tversky_index.score,
                   reference_fingerprint=reference,
                   query_fingerprint=query)