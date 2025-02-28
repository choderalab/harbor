import numpy as np
import logging
from typing import Optional, Union
import os


class FileLogger:
    """
    Gratefully copied from asapdiscovery.
    Thank you for your service Hugo MacDermott-Opeskin.
    """

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

    def getLogger(self) -> logging.Logger:
        return self.logger

    def set_level(self, level: int) -> None:
        self.logger.setLevel(level)
        self.handler.setLevel(level)


def get_ci_from_bootstraps(
    values: list[float], alpha: float = 0.95
) -> tuple[float, float]:
    """
    Calculate the confidence interval of a list of values
    """
    sorted_values = np.sort(values)
    n_values = len(sorted_values)
    lower_index = sorted_values[int(n_values * (1 - alpha))]
    upper_index = sorted_values[int(n_values * alpha)]
    return (lower_index, upper_index)
