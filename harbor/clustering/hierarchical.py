from harbor.similarity.mcss import get_mcs_mol, get_n_to_n_mcs
from openeye import oechem
from pydantic import Field, BaseModel
import numpy as np


class Cluster(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    cluster_id: str = Field(..., description="An id")
    children: list[str] = Field(..., description="Children")
    repr: oechem.OEMol
    height: int = Field(..., description="maximum number of layers above 0")

    @classmethod
    def from_mol(cls, layer_id, mol: oechem.OEMol) -> "Cluster":
        return cls(
            cluster_id=f"{0}_{layer_id}",
            repr=mol,
            children=[f"{mol.GetTitle()}"],
            height=0,
        )

    @classmethod
    def from_clusters(cls, layer_id, cluster1: "Cluster", cluster2: "Cluster"):
        height = max(cluster1.height, cluster2.height) + 1
        repr = get_mcs_mol(cluster1.repr, cluster2.repr)
        return cls(
            cluster_id=f"{height}_{layer_id}",
            children=[cluster1.cluster_id, cluster2.cluster_id],
            repr=repr,
            height=height,
        )


class ClusterResults(BaseModel):
    new: list[Cluster] = Field(..., description="Newly formed clusters")
    singles: list[Cluster] = Field(
        ..., description="Clusters which are out for this round"
    )
    outliers: list[Cluster] = Field(
        ...,
        description="Cluster centers further than the cutoff from any other molecule and therefore should be ignored for the rest of the clustering",
    )


def get_clusters_from_mcs_matrix(
    matrix: np.ndarray,
    clusters,
    cutoff,
) -> ClusterResults:
    """
    Get all pairs for which the maximum is reciprocal and is greater than
    """
    clusters = np.array(clusters)

    potential_match = np.argsort(matrix)[:, -2]
    max_mcs = np.sort(matrix)[:, -2]
    potential_match = np.array(potential_match, dtype="object")

    pairs = []
    singles = []
    outliers = []
    for i in range(len(potential_match)):
        if max_mcs[i] < cutoff:
            outliers.append(i)
            continue
        if np.isnan(potential_match[i]):
            continue
        j = potential_match[i]
        if np.isnan(potential_match[j]):
            continue
        if i == potential_match[j]:
            pairs.append((i, j))
            potential_match[j] = np.nan
        else:
            singles.append(i)
    new = [
        Cluster.from_clusters(i, clusters[j], clusters[k])
        for i, (j, k) in enumerate(pairs)
    ]
    return ClusterResults(
        new=new,
        singles=clusters[singles].tolist(),
        outliers=clusters[outliers].tolist(),
    )


def get_clusters_from_pairs(clusters, pairs):
    return [
        Cluster.from_clusters(i, clusters[j], clusters[k])
        for i, (j, k) in enumerate(pairs)
    ]


def mcs_wrapper(clusters: list[Cluster]):
    mols = [cluster.repr for cluster in clusters]
    return get_n_to_n_mcs(mols)
