import multiprocessing as mpi
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import harbor.analysis.cross_docking as cd
from pathlib import Path
import logging
from typing import Optional, Union
import os


# copied from asapdiscovery
class FileLogger:
    def __init__(
        self,
        logname: str,
        path: str,
        logfile: Optional[str] = None,
        level: Optional[Union[int, str]] = logging.DEBUG,
        format: Optional[
            str
        ] = "%(asctime)s | %(name)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s",
        stdout: Optional[bool] = False,
    ):
        self.name = logname
        self.logfile = logfile
        self.format = format
        self.level = level
        self.stdout = stdout

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)

        if self.logfile:
            self.handler = logging.FileHandler(
                os.path.join(path, self.logfile), mode="w"
            )
            self.handler.setLevel(self.level)
            self.formatter = logging.Formatter(self.format)
            self.handler.setFormatter(self.formatter)
            self.logger.addHandler(self.handler)

        # if self.stdout:
        #     console = Console()
        #     rich_handler = RichHandler(console=console)
        #     self.logger.addHandler(rich_handler)

    def getLogger(self) -> logging.Logger:
        return self.logger

    def set_level(self, level: int) -> None:
        self.logger.setLevel(level)
        self.handler.setLevel(level)


def get_args():
    parser = ArgumentParser(description="Run full cross docking evaluation")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the input csv file containing the cross docking data",
        required=True,
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        help="Path to the parameters yaml file",
        required=False,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output directory where the results will be stored",
        required=True,
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        help="Number of cpus to use for parallel processing",
        default=1,
    )
    return parser.parse_args()


def main():
    args = get_args()
    logger = FileLogger(
        logname="run_full_cross_docking_evaluation",
        path=args.output,
        logfile="run_full_cross_docking_evaluation.log",
    ).getLogger()
    args.output.mkdir(exist_ok=True, parents=True)
    output_dir = args.output
    logger.info("Reading input data")
    df = pd.read_csv(args.input)

    logger.info("Reading parameters")
    if args.parameters:
        settings = cd.Settings.from_yml_file(args.parameters)
    else:
        settings = cd.Settings()
    settings.to_yml_file(output_dir / "settings.yml")

    logger.info("Setting up evaluators")

    logger.info("Running cross docking evaluation")


if __name__ == "__main__":
    main()
