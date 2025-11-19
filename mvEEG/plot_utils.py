import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sista


def get_plot_line(a):
    """
    Takes in 2D array of shape [subjects,time points].
    Returns mean, and upper/lower SEM lines.
    """
    mean = a.mean(0)
    sem = sista.sem(a, 0)
    upper, lower = mean + sem, mean - sem
    return mean, upper, lower


def _plot_single_phase(ax, phase_time, ylim, hide=False, color="gray", title=None, fontsize=16):
    """
    plots stim bar and does type checking.
    Args:
        ax (matplotlib.axes.Axes): Axis to plot into.
        phase_time (iterable): Iterable of 2 ints.
        ylim (tuple): Y-axis limits of the figure.
        hide (bool, optional): Set to True to not actually plot the bar but return
                                a stim time. Defaults to False.
    """
    if hide:
        return

    assert len(ylim) == 2, "ylim should be a list of 2 floats"
    assert len(phase_time) == 2, "phase_time should be a list or tuple of 2 numbers"

    phase_lower = ylim[0]
    phase_upper = ylim[1]

    ax.fill_between(
        phase_time,
        [phase_lower, phase_lower],
        [phase_upper, phase_upper],
        color=color,
        alpha=0.5,
        zorder=-999,
    )
    if title is not None:
        ax.text(
            np.mean(phase_time),
            phase_upper * 0.95 + phase_lower * 0.05,
            title,
            fontsize=16,
            verticalalignment="top",
            horizontalalignment="center",
            color="black",
        )


def plot_trial_phases(ax, trial_phases, ylim, hide=False):
    """
    Plots bars for multiple phases.
    Args:
        ax (matplotlib.axes.Axes): Axis to plot into.
        trial_phases (dict): Dictionary of trial phases.
        ylim (tuple): Y-axis limits of the figure.
        hide (bool, optional): Set to True to not actually plot the bar but return
                                a stim time. Defaults to False.
    """
    if hide:
        return

    assert len(ylim) == 2, "ylim should be a list of 2 floats"

    if type(trial_phases) is dict:
        for phase, time in trial_phases.items():
            _plot_single_phase(ax, time, ylim, hide=hide, title=phase)
    elif type(trial_phases) in [list, tuple] and type(trial_phases[0]) in [list, tuple]:
        for phase_time in trial_phases:
            _plot_single_phase(ax, phase_time, ylim, hide=hide)
    else:
        raise TypeError(
            "trial_phases should either be a dictionary, or an iterable of lists or tuples. If adding a single phase, use  plot_single_phase."
        )


def pval_to_stars(p, thresholds={0.05: "*", 0.01: "**", 0.001: "***"}):
    """
    Converts p-value to a string of stars.

    Args:
        p (float): p-value to convert.
        thresholds (dict): Dictionary mapping p-value thresholds to star strings.

    Returns:
        str: Corresponding star string for the p-value.
    """
    for threshold, stars in sorted(thresholds.items()):
        if p < threshold:
            return stars
    return "n.s"  # Not significant
