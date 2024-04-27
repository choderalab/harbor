from harbor.similarity.mcss import get_mcs_mol, get_n_to_n_mcs
from openeye import oechem
from pydantic import Field, BaseModel, model_validator
from typing import Union
import numpy as np


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


def get_clusters_from_mcs_matrix(
    matrix: np.ndarray,
    clusters,
    cutoff,
    height: int = Field(..., description="maximum number of layers above 0"),
) -> ClusterResults:
    """
    Get all pairs for which the maximum is reciprocal and is greater than
    """
    clusters = np.array(clusters)
    print([cluster.cluster_id for cluster in clusters])

    # set the diagonal to 0
    np.fill_diagonal(matrix, 0)
    print(matrix)

    potential_match = np.argsort(matrix)[:, -1]

    duplicates = np.where(np.bincount(potential_match) > 1)[0]
    max_mcs = np.sort(matrix)[:, -1]
    print(potential_match)
    print(max_mcs)

    pairs = []
    singles = []
    outliers = []
    ignore = []
    for i in range(len(potential_match)):
        print(i)
        if max_mcs[i] < cutoff:
            outliers.append(i)
            continue

        j = potential_match[i]
        if i in ignore:
            print(i)
            continue
        if j in singles:
            singles.append(i)
            continue
        if j in duplicates:
            # pick a single pair to combine, and remove all others
            all_matches = np.where(potential_match == j)[0]
            print(i, j, all_matches)

            # for each reciprocal match, find the one with the highest mcs
            best_match = all_matches[np.argmax(max_mcs[all_matches])]
            if best_match == i:
                pairs.append((i, j))
                ignore.extend([i, j])
            else:
                singles.append(i)
        else:
            if potential_match[j] == i:
                pairs.append((i, j))
                ignore.extend([i, j])
            else:
                singles.append(i)
                ignore.append(i)
        print("Pairs", pairs)
        print("Singles", singles)
        print("Outliers", outliers)
        print("Ignore", ignore)
    print("Pairs", pairs)
    print("Singles", singles)
    print("Outliers", outliers)
    print("Ignore", ignore)

    new = [
        ClusterCenter.from_clusters(height, i, clusters[j], clusters[k])
        for i, (j, k) in enumerate(pairs)
    ]
    return ClusterResults(
        new=new,
        singles=clusters[singles].tolist(),
        outliers=clusters[outliers].tolist(),
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

            print("New clusters")
            print(
                [
                    child.cluster_id if isinstance(child, ClusterCenter) else child
                    for cluster in results.new
                    for child in cluster.children
                ]
            )

            print("Singles")
            print(
                [
                    child.cluster_id if isinstance(child, ClusterCenter) else child
                    for cluster in results.singles
                    for child in cluster.children
                ]
            )

            print("Outliers")
            print(
                [
                    child.cluster_id if isinstance(child, ClusterCenter) else child
                    for cluster in results.outliers
                    for child in cluster.children
                ]
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
