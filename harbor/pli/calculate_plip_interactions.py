from harbor.pli.plip_analysis_schema import PLIntReport
from pathlib import Path
import click
import yaml
from pebble import ProcessPool
from functools import partial
from concurrent.futures import TimeoutError


class ProcessingError(Exception):
    """Custom exception for processing errors"""

    pass


def analyze_structure(structure: Path, name: str, output_dir: Path) -> Path:
    """
    Analyze a single structure using PLIP.

    Parameters
    ----------
    structure : Path
        Path to the structure file
    name : str
        Name identifier for the output
    output_dir : Path
        Directory to save the output

    Returns
    -------
    Path
        Path to the output CSV file

    Raises
    ------
    ProcessingError
        If there's an error processing the structure
    """
    try:
        outpath = output_dir / f"{name}_{structure.stem}_interactions.csv"
        interactions = PLIntReport.from_complex_path(
            complex_path=structure,
        )
        interactions.to_csv(outpath)
        click.echo(f"Saved interactions to {outpath}")
        return outpath
    except Exception as e:
        error_msg = f"Error processing {structure}: {str(e)}"
        click.echo(error_msg, err=True)
        raise ProcessingError(error_msg)


def process_structure_batch(
    structures: list[Path], name: str, output_dir: Path, ncpus: int
) -> tuple[list[Path], list[str]]:
    """
    Process a batch of structures in parallel.

    Returns
    -------
    tuple[list[Path], list[str]]
        Lists of successful outputs and error messages
    """
    successful_outputs = []
    errors = []

    analyze_structure_partial = partial(
        analyze_structure,
        name=name,
        output_dir=output_dir,
    )

    with ProcessPool(max_workers=ncpus) as pool:
        future = pool.map(analyze_structure_partial, structures, timeout=300)
        iterator = future.result()

        while True:
            try:
                result = next(iterator)
                successful_outputs.append(result)
            except StopIteration:
                break
            except TimeoutError as e:
                errors.append(f"Processing timed out: {str(e)}")
            except ProcessingError as e:
                errors.append(str(e))
            except Exception as e:
                errors.append(f"Unexpected error: {str(e)}")

    return successful_outputs, errors


@click.command()
@click.option(
    "--pdb-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Path to directory containing PDB files",
    required=False,
)
@click.option(
    "--yaml-input",
    type=click.Path(exists=True, path_type=Path),
    help="Path to input yaml file containing name: path pairs",
    required=False,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./"),
    help="Path to output directory",
    required=False,
)
@click.option(
    "--ncpus", type=int, default=1, help="Number of cpus to use for parallel processing"
)
@click.option(
    "--error-log",
    type=click.Path(path_type=Path),
    help="Path to error log file",
    default="plip_errors.log",
)
def main(
    pdb_dir: Path, yaml_input: Path, output_dir: Path, ncpus: int, error_log: Path
):
    """
    Get PLIP interactions

    Basic usage, which create a csv file of the calculated interactions for all the pdb files in this directory:
    harbor calculate-plip-interactions --pdb-dir directory_with_pdb_files

    For more complex usage, you can provide a YAML file that maps names to directories containing PDB files:
    harbor calculate-plip-interactions --yaml-input input.yaml --output-dir output_directory --ncpus 4

    Where `input.yaml` contains a mapping of names to directories containing PDB files, and `output_directory` is where the interaction CSV files will be saved.

    i.e. input.yaml:
    ----------------
    crystal: 20250313_plip_analysis/crystal
    docked: 20250313_plip_analysis/docked
    """
    output_dir.mkdir(exist_ok=True)

    if not yaml_input and not pdb_dir:
        click.echo("Please provide either --pdb-dir or --yaml-input", err=True)
        raise click.Abort()

    all_errors = []
    if pdb_dir:
        input_dict = {"default": pdb_dir}
    elif yaml_input:
        try:
            with open(yaml_input, "r") as f:
                input_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            click.echo(f"Error reading YAML file: {e}", err=True)
            raise click.Abort()

    for name, structure_dir in input_dict.items():
        structure_dir = Path(structure_dir)
        if not structure_dir.exists():
            error_msg = f"Directory does not exist: {structure_dir}"
            all_errors.append(error_msg)
            click.echo(error_msg, err=True)
            continue

        click.echo(f"Loading all pdb structures in {structure_dir}")
        structures = list(structure_dir.glob("*.pdb"))

        if not structures:
            error_msg = f"No PDB files found in {structure_dir}"
            all_errors.append(error_msg)
            click.echo(error_msg, err=True)
            continue

        click.echo(f"Analyzing {len(structures)} structures")
        successful, errors = process_structure_batch(
            structures, name, output_dir, ncpus
        )

        if errors:
            all_errors.extend(errors)
            click.echo(
                f"Encountered {len(errors)} errors while processing {name}", err=True
            )

        click.echo(f"Successfully processed {len(successful)} structures for {name}")

    # Write error log if there were any errors
    if all_errors:
        with open(error_log, "w") as f:
            for error in all_errors:
                f.write(f"{error}\n")
        click.echo(f"Wrote {len(all_errors)} errors to {error_log}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
