"""
Calculate PLIP interaction scores between two structures.
Usually you would want to re-write this script to parallelize the analysis on your
"""

import pandas as pd

from harbor.pli.plip_analysis_schema import (
    PLIntReport,
    FingerprintLevel,
    InteractionScore,
)
from pathlib import Path
import click


@click.command()
@click.argument("reference", type=click.Path(exists=True, path_type=Path))
@click.argument("query", type=click.Path(exists=True, path_type=Path))
@click.argument(
    "-f",
    "--fingerprint-level",
    type=click.Choice([level.name for level in FingerprintLevel] + ["ALL"]),
    default="ALL",
    help="Level of fingerprint analysis to use.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./"),
    help="Output directory for analysis files.",
)
def main(reference: Path, query: Path, output_dir: Path, fingerprint_level: str):
    """Calculate PLIP interaction scores between reference and query structures.

    REFERENCE: Path to the reference PLIP interaction report CSV file.
    QUERY: Path to the query PLIP interaction report CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reports
    reference_report = PLIntReport.from_csv(reference)
    query_report = PLIntReport.from_csv(query)

    # Calculate scores
    scores = []
    if fingerprint_level == "ALL":
        for level in FingerprintLevel:
            scores.append(
                InteractionScore.from_fingerprints(
                    reference_report, query_report, level
                )
            )
    else:
        fingerprint_level = FingerprintLevel[fingerprint_level]
        scores.append(
            InteractionScore.from_fingerprints(
                reference_report, query_report, fingerprint_level
            )
        )
    df = pd.DataFrame.from_records(
        [
            {"Reference": reference.stem, "Query": query.stem, **score.dict()}
            for score in scores
        ]
    )

    # Save results
    output_file = output_dir / f"{reference.stem}_vs_{query.stem}_scores.csv"
    df.to_csv(output_file, index=False)
    click.echo(f"Scores saved to {output_file}")
