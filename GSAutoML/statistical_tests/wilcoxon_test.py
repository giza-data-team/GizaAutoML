from scipy.stats import wilcoxon
import pandas as pd


def wilcoxon_test(x: pd.Series, y: pd.Series = None, alpha: float = 0.05) -> bool:
    """A function that uses the Wilcoxon Signed-Rank test to see whether the difference between two pd.Series
    is significant or not
    :param x: Either the first set of measurements (in which case y is the second set of measurements),
     or the differences between two sets of measurements (in which case y is not to be specified.)
      Must be one-dimensional.
    :param y: Either the second set of measurements (if x is the first set of measurements), or not specified (if x
     is the differences between two sets of measurements.) Must be one-dimensional.
    :param alpha: Chosen significance level. Default: 0.05
    """
    statistic, p_value, _ = wilcoxon(x, y)

    # if the p value greater than significance level
    if p_value > alpha:
        # The difference is significant
        return True
    else:  # if the p value less than or equal significance level
        # The difference in insignificant
        return False
