import numpy as np
import mne_bids
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sista
import statsmodels as sm
from statsmodels.stats.multitest import multipletests
from .dataloader import DataLoader
from pathlib import Path
from collections import defaultdict

mpl.rcParams["font.sans-serif"] = "Arial"
mpl.rcParams["font.family"] = "sans-serif"
import os
import re
import pingouin as pg


class Interpreter:
    """
    A class to interpret and visualize EEG data for classification experiments.

    Arguments:
    ----------
    labels : list
        A list of labels corresponding to each value in the classification dataset
    data_dir: str
        The root directory of the dataset
    experiment_name: str
        the name of the experiment
    descriptions: list
        A list of descriptions (individual runs) to preload. Any descriptions not specified can still be lazy loaded when called
    subs: list
        A list of subjects to load. If not specified, will load all subjects with non-nan values


    Attributes:
    -----------
    labels : list
        A list of labels corresponding to each value in the classification dataset
    dataset : DataLoader
        An instance of DataLoader to load and manage the dataset. Initialized by the class
    colors : list
        A list of colors for plotting.

    Methods:
    --------
    get_plot_line():
        Takes in a 2D array of shape [subjects, time points] and returns mean, and upper/lower SEM lines.
    plot_stim_bar(ax, stim_time, ylim, hide=False):
        Plots a stimulus bar on the given axis and returns the aggregate stimulus time.
    do_significance_testing(t, a, b=0, test=None, alternative="two-sided", correction_method="fdr_bh"):
        Runs significance testing at each time point and determines the appropriate test.
    plot_acc(dset=None, ax=None, significance_testing=False, stim_time=[0, 200], save=False, title=None, ylim=[0, 1], chance_text_y=0.2, chance=0.5, skip_aesthetics=False, color="tab:red", sig_y=None, label=None):
        Plots accuracy for one subject for one condition.
    plot_hyperplane(labels, dset=None, ax=None, stim_time=[0, 200], title=None, ylim=[-4, 4], legend_title="Trial condition", legend_pos="lower right", label_text_x=-105, label_text_ys=[-3.4, 2.8], stim_label_xy=[100, 3.5], arrow_ys=[-1.1, 1.2], arrow_labels=None, significance_testing=False, sig_pairs=[(0, 1)], sig_ys=[0.2], alternatives=["greater"], sig_colors=["C0"]):
        Plots the hyperplane for the given labels and dataset.
    plot_hyperplane_contrast(dset=None, ax=None, pair=None, significance_testing=False, stim_time=[0, 200], title=None, ylim=[-1, 5], skip_aesthetics=False, color="tab:red", sig_y=None, label=None):
        Plots the hyperplane contrast for the given dataset and pair.
    plot_confusion_matrix(labels=None, dset=None, ax=None, earliest_t=200, lower=0, upper=1, chance=None, color_map=plt.cm.RdGy_r, title=""):
        Plots the confusion matrix for the classifier.
    plot_2_contrasts(dsets, pairs, labels, colors=["C0", "C1"], ax=None, significance_between=False, sig_y_between=3, sig_color="C2", test_tail="two-sided", sig_ys=[-0.5, -0.6], **kwargs):
        Plots two hyperplane contrast graphs on the axis for comparison.
    plot_2_contrasts_bar(dsets, pairs, labels, t_start=200, ax=None, significance_between=False, sig_y=1, test_tail="two-sided", ylim=None, title=None, **kwargs):
        Plots two hyperplane contrasts as a bar graph and computes the Bayes factor between the two.
    """
    
    def __init__(
        self,
        labels: list,
        data_dir: str,
        experiment_name: str,
        descriptions: list = [],
        subs: list = [],
    ):
        
        self.dataset = DataLoader(root_dir=data_dir,
                                  data_type="classification",
                                  experiment_name=experiment_name,
                                  descriptions=descriptions,
                                  subs=subs)
        
        self.labels = labels
        self.colors = ["royalblue", "firebrick", "forestgreen", "orange", "purple"]

    @staticmethod
    def get_plot_line(a):
        """
        Takes in 2D array of shape [subjects,time points].
        Returns mean, and upper/lower SEM lines.
        """
        mean = a.mean(0)
        sem = sista.sem(a, 0)
        upper, lower = mean + sem, mean - sem
        return mean, upper, lower

    @staticmethod
    def plot_stim_bar(ax, stim_time, ylim, hide=False):

        """
        plots stim bar and does type checking.
        Also returns an aggregate stim_time of the whole stim period
        Args:
            ax (matplotlib.axes.Axes): Axis to plot into.
            stim_time (iterable): Iterable of 2 ints (for a single stim bar), or 
                                  iterable of iterables of 2 ints (for multiple bars).
            ylim (tuple): Y-axis limits of the figure.
            hide (bool, optional): Set to True to not actually plot the bar but return 
                                   a stim time. Defaults to False.
        """

        stim_lower = ylim[0] + 0.01
        stim_upper = ylim[1]
        if type(stim_time[0]) is int:
            if not hide:
                ax.fill_between(
                    stim_time,
                    [stim_lower, stim_lower],
                    [stim_upper, stim_upper],
                    color="gray",
                    alpha=0.5,
                    zorder=-999,
                )
            return stim_time
        elif type(stim_time[0]) is list or type(stim_time[0]) is tuple:
            for time in stim_time:
                if not hide:
                    ax.fill_between(
                        time,
                        [stim_lower, stim_lower],
                        [stim_upper, stim_upper],
                        color="gray",
                        alpha=0.5,
                        zorder=-999,
                    )
            return [stim_time[0][0], stim_time[-1][1]]
        else:
            raise TypeError(
                "stim_time should either be an iterable of lists or tuples, or a single iterable of ints"
            )

    @staticmethod
    def do_significance_testing(
        t, a, b=0, test=None, alternative="two-sided", correction_method="fdr_bh"
    ):
        """
        Perform significance testing on the provided data.
        Parameters:
        t (array-like): Time points corresponding to the data.
        a (array-like): Data array to test.
        b (array-like, optional): Baseline or comparison data. Default is 0.
        test (callable, optional): Statistical test function to use. Default is None.
        alternative (str, optional): Defines the alternative hypothesis. Options are "two-sided", "less", or "greater". Default is "two-sided".
        correction_method (str, optional): Method for multiple testing correction. Default is "fdr_bh".
        Returns:
        A tuple containing:
            - p (array-like): p-values after testing and correction.
            - sig05 (array-like): Boolean array indicating significance at the 0.05 level.
        """


        if test is None:
            if type(b) == int or type(b) == float:
                test = sista.ttest_1samp
            elif a.size == b.size:
                test = sista.ttest_rel

        _, p = test(a, b, alternative=alternative)
        p = p[t > 0]  # only select out times > 0
        if correction_method is not None:
            _, p, _, _ = multipletests(p, alpha=0.05, method=correction_method)

        sig05 = p < 0.05

        return p, sig05



    def plot_acc(
        self,
        dset=None,
        ax=None,
        significance_testing=False,
        stim_time=[0, 200],
        save=False,
        title=None,
        ylim=[0, 1],
        chance_text_y=0.2,
        chance=0.5,
        skip_aesthetics=False,
        color="tab:red",
        sig_y=None,
        label=None,
    ):
        """
        Plots accuracy for one subject for one condition
        Arguments:
        dset (str): description of the condition to plot (can leave blank if only one condition)
        ax (matplotlib axis): axis to plot into (creates if doesn't exist)
        significance_testing (bool): run significance testing and plot significance dots
        stim_time (list of 2 times, or list of stim_times): time of stimulus presentation
        save (bool): whether to save figure
        title (str): title of figure
        ylim (list of 2 floats): y limits of figure
        chance_text_y (float): y position of chance label
        chance (float): chance level for plot (defaults to 0.5)
        skip_aesthetics (bool): set to True to skip aesthetic features and only plot the line
        color (str): color of line
        sig_y (float): y position of significance dots
        label (str): label of line
        """

        # extract values and average over iterations

        acc, acc_shuff, t = self.dataset.get_data(
            dset, keys=["accuracy", "shuffledAccuracy", "times"]
        )

        acc = acc.mean(1)
        acc_shuff = acc_shuff.mean(1)

        acc_mean, acc_upper, acc_lower = self.get_plot_line(acc)
        acc_shuff_mean, acc_shuff_upper, acc_shuff_lower = self.get_plot_line(acc_shuff)

        sig_y = chance - 0.05 if sig_y is None else sig_y

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(t, acc_mean, color=color, label=label, linewidth=2)
        ax.fill_between(t, acc_upper, acc_lower, color=color, alpha=0.5)

        ax.plot(t, acc_shuff_mean, color="gray")
        ax.fill_between(t, acc_shuff_upper, acc_shuff_lower, color="gray", alpha=0.5)

        if significance_testing:
            p, sig05 = self.do_significance_testing(
                t, acc, acc_shuff, alternative="greater"
            )
            ax.scatter(
                t[t > 0][sig05],
                np.full(sum(sig05), sig_y),
                color=color,
                s=10,
                marker="s",
                zorder=999,
            )
            print(
                f"% timepoints significant: {round(sum(sig05)/len(sig05)*100,2)} ({sum(sig05)}/{len(sig05)})%"
            )

        if not skip_aesthetics:
            stim_time = self.plot_stim_bar(ax, stim_time, ylim)
            ax.plot(t, np.ones((len(t))) * chance, "--", color="gray", zorder=0)
            # aesthetics
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks(np.arange(0.1, 1.1, 0.1))
            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)
            plt.xlim(min(t), max(t))
            plt.ylim(ylim)
            # ax.legend(loc="lower right", frameon=False, fontsize=11)

            # labelling
            ax.set_xlabel("Time from stimulus onset (ms)", fontsize=14)
            ax.set_ylabel("Classification accuracy", fontsize=14)
            ax.text(
                0.17,
                0.9,
                "Stim",
                transform=ax.transAxes,
                fontsize=16,
                verticalalignment="top",
                color="black",
            )
            if title is not None:
                ax.set_title(title, fontsize=18)
        return ax

    def plot_hyperplane(

        self,
        labels,
        dset=None,
        ax=None,
        stim_time=[0, 200],
        title=None,
        ylim=[-4, 4],
        legend_title="Trial condition",
        legend_pos="lower right",
        label_text_x=-105,
        label_text_ys=[-3.4, 2.8],
        stim_label_xy=[100, 3.5],
        arrow_ys=[-1.1, 1.2],
        arrow_labels=None,
        significance_testing=False,
        sig_pairs=[(0, 1)],
        sig_ys=[0.2],
        alternatives=["greater"],
        sig_colors=["C0"],
    ):

        """
        Plots the hyperplane for the given dataset and labels.
        Parameters:
        -----------
        labels : list
            List of condition labels to plot.
        dset : str, optional
            Dataset identifier to retrieve data from. Default is None.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, a new figure and axes are created. Default is None.
        stim_time : list, optional
            Time range for the stimulus bar. Default is [0, 200].
        title : str, optional
            Title of the plot. Default is None.
        ylim : list, optional
            Y-axis limits for the plot. Default is [-4, 4].
        legend_title : str, optional
            Title for the legend. Default is "Trial condition".
        legend_pos : str, optional
            Position of the legend. Default is "lower right".
        label_text_x : int, optional
            X-coordinate for the label text. Default is -105.
        label_text_ys : list, optional
            Y-coordinates for the label text. Default is [-3.4, 2.8].
        stim_label_xy : list, optional
            Coordinates for the stimulus label. Default is [100, 3.5].
        arrow_ys : list, optional
            Y-coordinates for the arrows. Default is [-1.1, 1.2].
        arrow_labels : list, optional
            Labels for the arrows. If None, defaults to the first two labels. Default is None.
        significance_testing : bool, optional
            Whether to perform significance testing. Default is False.
        sig_pairs : list, optional
            Pairs of conditions to test for significance. Default is [(0, 1)].
        sig_ys : list, optional
            Y-coordinates for significance markers. Default is [0.2].
        alternatives : list, optional
            List of alternative hypotheses for significance testing. Default is ["greater"].
        sig_colors : list, optional
            Colors for significance markers. Default is ["C0"].

        """

        confidence_scores, t = self.dataset.get_data(dset, keys=["confidenceScores", "times"])

        confidence_scores = confidence_scores.mean(1)

        if ax is None:
            _, ax = plt.subplots()

        stim_time = self.plot_stim_bar(ax=ax, stim_time=stim_time, ylim=ylim)
        ax.plot(t, np.zeros((len(t))), "--", color="gray")

        condition_subset = [self.labels.index(label) for label in labels]
        if len(condition_subset) == 0:
            raise ValueError(f"No conditions were selected from {self.labels}")

        for i,condition in enumerate(condition_subset):
            mean, upper, lower = self.get_plot_line(confidence_scores[:, condition])
            ax.plot(t, mean, color=self.colors[i], label=self.labels[i], linewidth=2)
            ax.fill_between(t, upper, lower, color=self.colors[i], alpha=0.5)

        leg = plt.legend(title=legend_title, loc=legend_pos, fontsize=12)
        plt.setp(leg.get_title(), fontsize=12)

        # aesthetics
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        plt.setp(ax.get_xticklabels(), fontsize=14)

        plt.ylim(ylim)

        if significance_testing:
            sig_pairs = [self._get_pair_from_label(pair) for pair in sig_pairs]
            if len(alternatives) < len(sig_pairs):
                alternatives = alternatives + [[alternatives[0]] * (len(sig_pairs) - len(alternatives))][0]
            if len(sig_colors) < len(sig_pairs):
                sig_colors = sig_colors + [f'C{i}' for i in range(len(sig_colors),len(sig_pairs))]
            if len(sig_ys) < len(sig_pairs):
                sig_ys = sig_ys + [sig_ys[-1] + 0.2 * i for i in range(len(sig_pairs)-len(sig_ys))]

            for pair, alternative, color, y in zip(
                sig_pairs, alternatives, sig_colors, sig_ys
            ):
                p, sig05 = self.do_significance_testing(
                    t,
                    confidence_scores[:, pair[0]],
                    confidence_scores[:, pair[1]],
                    alternative=alternative,
                )
                ax.scatter(
                    t[t > 0][sig05],
                    np.full(sum(sig05), y),
                    color=color,
                    s=10,
                    marker="s",
                    zorder=999,
                )
                print(
                    f"% timepoints significant for {labels[pair[0]]} vs {labels[pair[1]]} (alternative = {alternative}): {round(sum(sig05)/len(sig05)*100,2)} ({sum(sig05)}/{len(sig05)})%"
                )

        plt.title(title, fontsize=18)
        plt.xlabel("Time from stimulus onset (ms)", fontsize=14)
        plt.ylabel("Distance from hyperplane (a.u.)", fontsize=14)
        if arrow_labels is None:
            arrow_labels = [labels[0], labels[1]]

        plt.text(
            label_text_x,
            label_text_ys[0],
            f"Predicted\n{arrow_labels[0]}",
            fontsize=12,
            ha="center",
        )
        plt.text(
            label_text_x,
            label_text_ys[1],
            f"Predicted\n{arrow_labels[1]}",
            fontsize=12,
            ha="center",
        )
        plt.text(
            stim_label_xy[0], stim_label_xy[1], "Stim", fontsize=14, ha="center", c="k"
        )
        plt.arrow(
            label_text_x, arrow_ys[0], 0, -1, head_width=45, head_length=0.25, color="k"
        )
        plt.arrow(
            label_text_x, arrow_ys[1], 0, 1, head_width=45, head_length=0.25, color="k"
        )


    def _get_pair_from_label(self,pair):
        """
        Helper function to translate labeled pairs into numerical ones
        """

        if pair is None:
            pair = [0, 1]
        elif all([type(p) == int for p in pair]):
            pair = pair
        elif all([type(p) == str for p in pair]):
            pair = [self.labels.index(p) for p in pair]
        else:
            raise ValueError("Invalid Conditions. Must be None, a list of ints, or a list of str")

        return pair

    def plot_hyperplane_contrast(
        self,
        dset=None,
        ax=None,
        pair=None,
        significance_testing=False,
        stim_time=[0, 200],
        title=None,
        ylim=[-1, 5],
        skip_aesthetics=False,
        color="tab:red",
        sig_y=None,
        label=None,
    ):
        """
        Plots accuracy for one subject for one condition
        Arguments:
        dset (str): description of the condition to plot (can leave blank if only one condition)
        ax (matplotlib axis): axis to plot into (creates if doesn't exist)
        pair: which pair to calculate
        significance_testing (bool): run significance testing and plot significance dots
        stim_time (list of 2 times, or list of stim_times): time of stimulus presentation
        save (bool): whether to save figure
        title (str): title of figure
        ylim (list of 2 floats): y limits of figure
        skip_aesthetics (bool): set to True to skip aesthetic features and only plot the line
        color (str): color of line
        sig_y (float): y position of significance dots
        label (str): label of line
        """

        pair = self._get_pair_from_label(pair)

        confidence_scores, t = self.dataset.get_data(dset, keys=["confidenceScores", "times"])
        contrast = np.mean(
            confidence_scores[:, :, pair[1]] - confidence_scores[:, :, pair[0]], axis=1
        )

        mean, upper, lower = self.get_plot_line(contrast)

        sig_y = -0.05 if sig_y is None else sig_y

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(t, mean, color=color, label=label, linewidth=2)
        ax.fill_between(t, upper, lower, color=color, alpha=0.5)

        if significance_testing:
            p, sig05 = self.do_significance_testing(
                t, contrast, 0, alternative="greater"
            )
            ax.scatter(
                t[t > 0][sig05],
                np.full(sum(sig05), sig_y),
                color=color,
                s=10,
                marker="s",
                zorder=999,
            )
            print(
                f"% timepoints significant: {round(sum(sig05)/len(sig05)*100,2)} ({sum(sig05)}/{len(sig05)})%"
            )

        if not skip_aesthetics:
            stim_time = self.plot_stim_bar(ax, stim_time, ylim)
            ax.plot(t, np.zeros((len(t))), "--", color="gray", zorder=0)
            # aesthetics
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")
            # ax.yaxis.set_ticks(np.arange(.1, 1.1, .1))
            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)
            plt.xlim(min(t), max(t))
            plt.ylim(ylim)
            ax.legend(loc="lower right", frameon=False, fontsize=11)

            # labelling
            ax.set_xlabel("Time from stimulus onset (ms)", fontsize=14)
            ax.set_ylabel("Hyperplane Contrast (a.u.)", fontsize=14)
            ax.text(
                0.17,
                0.9,
                "Stim",
                transform=ax.transAxes,
                fontsize=16,
                verticalalignment="top",
                color="black",
            )
            if title is not None:
                ax.set_title(title, fontsize=18)
        return ax

    def plot_confusion_matrix(
        self,
        labels=None,
        dset=None,
        ax=None,
        earliest_t=200,
        lower=0,
        upper=1,
        chance=None,
        color_map=plt.cm.RdGy_r,
        title="",
    ):
        """
        plots the confusion matrix for the classifier

        Arguments:
        dset (str): description of the condition to plot (can leave blank if only one condition)
        labels (list of str): labels for the confusion matrix
        ax (matplotlib axis): axis to plot into (creates if doesn't exist)
        earliest_t (int): time to start plotting from (should be start of delay period)\
        lower (float): lower bound of color map
        upper (float): upper bound of color map
        chance (float): chance level for plot (defaults to 0.5)
        color_map (matplotlib color map): color map for plot
        title (str): title of plot
        """

        conf_mat, t = self.dataset.get_data(dset, keys=["confusionMatrix", "times"])

        if ax is None:
            _, ax = plt.subplots()

        cm = np.mean(conf_mat[..., t > earliest_t], axis=(0, 1, -1))

        # Normalize
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Get labels and chance level, if necessary
        if labels is None:
            labels = self.labels

        if chance is None:
            chance = (upper - lower) / cm.shape[0]

        # Generate plot
        ax = sns.heatmap(
            cm,
            center=chance,
            vmin=lower,
            vmax=upper,
            xticklabels=labels,
            yticklabels=labels,
            # non-arg aesthetics
            annot=True,
            square=True,
            annot_kws={"fontsize": 16},
            linewidths=0.5,
            cmap=color_map,
            ax=ax,
        )

        # Clean up axes
        plt.ylabel("True Label", fontsize=16)
        plt.title("Predicted Label", fontsize=16)
        plt.yticks(rotation=0)
        plt.tick_params(
            axis="both",
            which="major",
            labelsize=15,
            labelbottom=False,
            bottom=False,
            top=False,
            labeltop=True,
            left=False,
        )
        plt.suptitle(title)

        plt.tight_layout()
        return ax

    ## functions to plot multiple of something

    def plot_2_contrasts(
        self,
        dsets,
        pairs,
        labels,
        colors=["C0", "C1"],
        ax=None,
        significance_between=False,
        sig_y_between=3,
        sig_color="C2",
        test_tail="two-sided",
        sig_ys = [-0.5,-0.6],
        **kwargs,
    ):
        
        """
        Plots two hyperplane contrast graphs on the axis. Useful for comparing crosstraining.

        Wrapper for plot_hyperplane_contrast

        Arguments:
        dsets (list of 2 str): descriptions of the conditions to plot. One per line
        pairs (list of 2 tuples): pairs of conditions to compare. eg: [("C2",'C1"),("O2","O1")]. 
        labels (list of 2 str): labels for the legend
        colors (list of 2 str): colors for each line
        ax (matplotlib axis): axis to plot into (creates if doesn't exist)
        significance_between (bool): run significance testing between the two contrasts
        sig_y_between (float): y position of significance dots
        sig_color (str): color of significance dots
        test_tail (str): tail of the test to run
        sig_ys (list of 2 floats): y position of significance dots for 
        Other kwargs are passed to plot_hyperplane_contrast


        """

        if ax is None:
            _, ax = plt.subplots()

        custom_lines = []  # for the legend
        for color in colors:
            custom_lines.append(mpl.lines.Line2D([0], [0], color=color, lw=4))

        self.plot_hyperplane_contrast(
            dset=dsets[0],
            color=colors[0],
            pair=pairs[0],
            label=labels[0],
            ax=ax,
            sig_y = sig_ys[0],
            **kwargs,
        )
        self.plot_hyperplane_contrast(
            dset=dsets[1],
            color=colors[1],
            pair=pairs[1],
            label=labels[1],
            ax=ax,
            skip_aesthetics=True,
            sig_y = sig_ys[1],
            **kwargs,
        )
        ax.legend(custom_lines, labels, loc="lower right", frameon=True, fontsize=11)

        if significance_between:
            cs1, t = self.dataset.get_data(dsets[0], keys=["confidenceScores", "times"])
            cs2 = self.dataset.get_data(dsets[1], keys=["confidenceScores"])
            pair0 = self._get_pair_from_label(pairs[0])
            pair1 = self._get_pair_from_label(pairs[1])

            contrast_1 = np.mean(
                cs1[:, :, pair0[1]] - cs1[:, :, pair0[0]], axis=1
            )
            contrast_2 = np.mean(
                cs2[:, :, pair1[1]] - cs2[:, :, pair1[0]], axis=1
            )

            p, sig05 = self.do_significance_testing(
                t, contrast_1, contrast_2, alternative=test_tail
            )
            ax.scatter(
                t[t > 0][sig05],
                np.full(sum(sig05), sig_y_between),
                color=sig_color,
                s=10,
                marker="s",
                zorder=999,
            )

        return ax

    def plot_2_contrasts_bar(
        self,
        dsets,
        pairs,
        labels,
        t_start=200,
        ax=None,
        significance_between=False,
        sig_y=1,
        test_tail="two-sided",
        ylim=None,
        title=None,
        **kwargs,
    ):
        
        """
        Plots two hyperplane contrasts as a bar graph. Also computes the bayes-factor between the two.

        Arguments:
        dsets (list of 2 str): descriptions of the conditions to plot. One per line
        pairs (list of 2 tuples): pairs of conditions to compare.
          eg: [("C2",'C1"),("O2","O1")].
          The second will be subtracted from the first
        labels (list of 2 str): labels for the legend
        t_start (int): time to start calculating the contrast from
        ax (matplotlib axis): axis to plot into (creates if doesn't exist)
        significance_between (bool): run significance testing between the two contrasts?
        sig_y (float): y position of significance dots
        test_tail (str): tail of the test to run (two-sided, greater, less)
        ylim (list of 2 floats): y limits of figure
        title (str): title of figure


        """

        if ax is None:
            _, ax = plt.subplots()

        cs1, t = self.dataset.get_data(dsets[0], keys=["confidenceScores", "times"])
        cs2 = self.dataset.get_data(dsets[1], keys=["confidenceScores"])
        pair0 = self._get_pair_from_label(pairs[0])
        pair1 = self._get_pair_from_label(pairs[1])

        contrasts = [
            np.mean(cs1[:, :, pair0[1]][..., t > t_start] 
                    - cs1[:, :, pair0[0]][..., t > t_start],
                    axis=(1,2)),
            np.mean(cs2[:, :, pair1[1]][..., t > t_start]
                    - cs2[:, :, pair1[0]][..., t > t_start],
                    axis=(1,2))
        ]


        ax = sns.barplot(data=contrasts, errorbar="se", ax=ax, **kwargs)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.hlines(0, -0.5, 1.5, "gray", "--")
        ax.set_ylabel("Hyperplane Contrast (a.u.)")
        ax.set_xticks([0, 1], labels)

        plt.tight_layout()

        stats = pg.ttest(contrasts[0], contrasts[1], paired=True, alternative=test_tail)
        p = stats["p-val"].values[0]



        if significance_between:
            ax.plot([0, 1], [sig_y, sig_y], "k")

            if p > 0.05:
                stars = "n.s."
            elif p > 0.01:
                stars = "*"
            elif p > 0.001:
                stars = "**"
            else:
                stars = "***"
            ax.set_title(title)
            ax.text(
                0.5, sig_y + 0.05, f'{stars}\nBF10 = {stats["BF10"].values[0]}', ha="center"
            )
            print(stats)
