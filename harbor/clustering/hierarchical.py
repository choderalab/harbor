from harbor.similarity.mcss import get_mcs_mol, get_n_to_n_mcs
from openeye import oechem
from pydantic import Field, BaseModel, model_validator
from typing import Union
import numpy as np
from enum import Enum


class ClusterCenter(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    cluster_id: str = Field(..., description="An id")
    children: list[Union[str, "ClusterCenter"]] = Field(..., description="Children")
    repr: oechem.OEMol
    height: int = Field(..., description="maximum number of layers above 0")

    @classmethod
    def from_mol(cls, layer_id, mol: oechem.OEMol) -> "ClusterCenter":
        return cls(
            cluster_id=f"{0}_{layer_id}",
            repr=mol,
            children=[f"{layer_id}"],
            height=0,
        )

    @classmethod
    def from_clusters(
        cls, height, layer_id, cluster1: "ClusterCenter", cluster2: "ClusterCenter"
    ):
        repr = get_mcs_mol(cluster1.repr, cluster2.repr)
        return cls(
            cluster_id=f"{height}_{layer_id}",
            children=[cluster1, cluster2],
            repr=repr,
            height=height,
        )


class ClusterResults(BaseModel):
    new: list[ClusterCenter] = Field(..., description="Newly formed clusters")
    singles: list[ClusterCenter] = Field(
        ..., description="Cluster centers which are out for this round"
    )
    outliers: list[ClusterCenter] = Field(
        ...,
        description="Cluster centers further than the cutoff from any other molecule "
        "and therefore should be ignored for the rest of the clustering",
    )

class ClusterStatus(Enum):
    unassigned = 0
    success = 1
    single = 2
    outlier = 3


def get_clusters_from_mcs_matrix(
    mtx: np.ndarray,
    clusters,
    cutoff,
    height: int = Field(..., description="maximum number of layers above 0"),
) -> ClusterResults:
    """
    Get all pairs for which the maximum is reciprocal and is greater than
    """
    clusters = np.array(clusters)
    print([cluster.cluster_id for cluster in clusters])

    # set the diagonal to 0 so we don't match with ourself
    np.fill_diagonal(mtx, 0)
    status_array = np.array([ClusterStatus.unassigned]*len(clusters))
    print(mtx)

    pairs = []
    for i, cluster in enumerate(clusters):

        # skip if already assigned
        if status_array[i] != ClusterStatus.unassigned:
            continue

        # get largest mcs
        mcs_array = mtx[i]
        max_mcs = mcs_array.max()

        # if max mcs < cutoff, add to outliers
        if max_mcs < cutoff:
            status_array[i] = ClusterStatus.outlier

        # get all potential matches
        potential_matches = np.where(mcs_array == max_mcs)[0]
        print(i, potential_matches)
        
        # use to break if we find a match
        fail_to_find = True
        for j in potential_matches:

            # potential match can't have already been assigned
            if status_array[j] != ClusterStatus.unassigned:
                continue

            # get the max mcs of the potential match
            mcs_array = mtx[j]
            sweetheart_max_mcs = mcs_array.max()
            
            # for reasons unknown to me, the MCS isn't actually reciprocal
            reciprocal_mcs = mtx[j, i]

            print(i, j, max_mcs, sweetheart_max_mcs, reciprocal_mcs)


            # only match if the max mcs of the potential match is <= to our mcs
            # with them
            if max_mcs >= sweetheart_max_mcs or reciprocal_mcs >= sweetheart_max_mcs:
                fail_to_find = False
                break
        
        if fail_to_find:
            status_array[i] = ClusterStatus.single
        else:
            pairs.append((i, j))
            status_array[i] = ClusterStatus.success
            status_array[j] = ClusterStatus.success
    print(status_array)
    new = [
        ClusterCenter.from_clusters(height, i, clusters[j], clusters[k])
        for i, (j, k) in enumerate(pairs)
    ]
    singles = clusters[status_array == ClusterStatus.single].tolist()
    outliers = clusters[status_array == ClusterStatus.outlier].tolist()
    return ClusterResults(
        new=new,
        singles=singles,
        outliers=outliers,
    )


def get_clusters_from_pairs(clusters, pairs):
    return [
        ClusterCenter.from_clusters(i, clusters[j], clusters[k])
        for i, (j, k) in enumerate(pairs)
    ]


def mcs_wrapper(clusters: list[ClusterCenter]):
    mols = [cluster.repr for cluster in clusters]
    return get_n_to_n_mcs(mols)


class HeirarchicalClustering(BaseModel):
    """
    A class to run and process heirarchical clustering for molecules.
    """

    molecules: list[oechem.OEMol] = Field(
        ..., description="A list of molecules to cluster"
    )
    mol_ids: list[str] = Field(description="Molecule IDs")

    class Config:
        arbitrary_types_allowed = True

    from pydantic import field_validator, model_validator

    @model_validator(mode="before")
    def check_lengths_match(cls, values):
        mol_ids = values["mol_ids"]
        molecules = values["molecules"]

        if not mol_ids:
            mol_ids = range(len(molecules))

        if not len(mol_ids) == len(molecules):
            raise ValueError(
                f"Length of mol_ids ({len(mol_ids)}) does not match molecules ({len(molecules)}"
            )
        return values

    @property
    def num_mols(self):
        return len(self.molecules)

    def cluster(self, max_iterations: int = 10, cutoff: int = 12):

        # Make initial clusters
        clusters = [
            ClusterCenter.from_mol(mol_id, mol)
            for mol_id, mol in zip(self.mol_ids, self.molecules)
        ]

        # keep track of the molecule outliers
        cluster_records = {}

        # main clustering loop
        i = 0
        while i <= max_iterations:
            i += 1
            # generate n x n matrix of MCS values
            mcs_matrix = mcs_wrapper(clusters)

            # get new cluster centers
            results = get_clusters_from_mcs_matrix(
                mcs_matrix, clusters, cutoff=cutoff, height=i
            )

            # update clusters with the new cluster centers and the previously unmatched singles
            clusters = results.new + results.singles

            cluster_records.update(
                {cluster.cluster_id: cluster for cluster in results.outliers}
            )
            if len(results.singles) == 0:
                cluster_records.update(
                    {cluster.cluster_id: cluster for cluster in results.new}
                )
                break
        return cluster_records


class HierarchicalClusteringResults(BaseModel):
    """
    A class to store the results of hierarchical clustering.
    """

    clusters: list[ClusterCenter] = Field(..., description="Cluster centers")
    outliers: list[ClusterCenter] = Field(
        ...,
        description="Molecules different enough from the rest that they were excluded by the cutoff",
    )
    clusterer: HeirarchicalClustering = Field(..., description="The clusterer used")
    cutoff: int = Field(..., description="The cutoff used for clustering")
    max_iterations: int = Field(..., description="The maximum number of iterations")
