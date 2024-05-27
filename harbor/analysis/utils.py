import numpy as np


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
