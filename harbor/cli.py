import click
from harbor.pli.calculate_plip_score import main as calculate_plip_score
from harbor.pli.calculate_plip_interactions import main as calculate_plip_interactions


@click.group(help="Command-line interface for Harbor")
def cli():
    pass


cli.add_command(calculate_plip_score, name="calculate-plip-score")
cli.add_command(calculate_plip_interactions, name="calculate-plip-interactions")
