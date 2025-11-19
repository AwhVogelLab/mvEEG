import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sista
from statsmodels.stats.multitest import multipletests
from .dataloader import DataLoader
from .plot_utils import get_plot_line, plot_trial_phases, pval_to_stars
import pingouin as pg
from mne.stats import permutation_cluster_1samp_test

mpl.rcParams["font.sans-serif"] = "Arial"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["svg.fonttype"] = "none"


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
    plot_single_phase(ax, phase_time, ylim, hide=False):
        Plots a shaded region on the given axis and returns the aggregate stimulus time.
    do_significance_testing(t, a, b=0, test=None, alternative="two-sided", correction_method="fdr_bh"):
        Runs significance testing at each time point and determines the appropriate test.
    plot_acc(dset=None, ax=None, significance_testing=False, phase_times=None, save=False, title=None, ylim=[0, 1], chance_text_y=0.2, chance=0.5, skip_aesthetics=False, color="tab:red", sig_y=None, label=None):
        Plots accuracy for one subject for one condition.
    plot_hyperplane(labels, dset=None, ax=None, phase_times=None, title=None, ylim=[-4, 4], legend_title="Trial condition", legend_pos="lower right", label_text_x=-105, label_text_ys=[-3.4, 2.8], stim_label_xy=[100, 3.5], arrow_ys=[-1.1, 1.2], arrow_labels=None, significance_testing=False, sig_pairs=[(0, 1)], sig_ys=[0.2], alternatives=["greater"], sig_colors=["C0"]):
        Plots the hyperplane for the given labels and dataset.
    plot_hyperplane_contrast(dset=None, ax=None, pair=None, significance_testing=False, phase_times=None, title=None, ylim=[-1, 5], skip_aesthetics=False, color="tab:red", sig_y=None, label=None):
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
        data_type: str = "classification",
        descriptions: list | None = None,
        subs: list | None = None,
        trial_phases: dict | None = None,
    ):

        self.dataset = DataLoader(
            root_dir=data_dir,
            data_type=data_type,
            experiment_name=experiment_name,
            descriptions=descriptions,
            subs=subs,
        )

        self.labels = labels
        self.colors = ["royalblue", "firebrick", "forestgreen", "orange", "purple", "cyan"]
        self.trial_phases = trial_phases

    @staticmethod
    def do_significance_testing(t, a, b=0, test=None, alternative="two-sided", correction_method="fdr_bh"):
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
        save=False,
        title=None,
        ylim=[0, 1],
        chance_text_y=0.2,
        chance=0.5,
        skip_aesthetics=False,
        color="tab:red",
        sig_y=None,
        label=None,
        trial_phases=None,
    ):
        """
        Plots accuracy for one subject for one condition
        Arguments:
        dset (str): description of the condition to plot (can leave blank if only one condition)
        ax (matplotlib axis): axis to plot into (creates if doesn't exist)
        significance_testing (bool): run significance testing and plot significance dots
        phase_times: dictionary or list of lists of phase times
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

        acc, acc_shuff, t = self.dataset.get_data(dset, keys=["accuracy", "shuffledAccuracy", "times"])

        acc = acc.mean(1)
        acc_shuff = acc_shuff.mean(1)

        acc_mean, acc_upper, acc_lower = get_plot_line(acc)
        acc_shuff_mean, acc_shuff_upper, acc_shuff_lower = get_plot_line(acc_shuff)

        sig_y = chance - 0.05 if sig_y is None else sig_y

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(t, acc_mean, color=color, label=label, linewidth=2)
        ax.fill_between(t, acc_upper, acc_lower, color=color, alpha=0.5)

        ax.plot(t, acc_shuff_mean, color="gray")
        ax.fill_between(t, acc_shuff_upper, acc_shuff_lower, color="gray", alpha=0.5)

        if significance_testing:
            p, sig05 = self.do_significance_testing(t, acc, acc_shuff, alternative="greater")
            ax.scatter(
                t[t > 0][sig05],
                np.full(sum(sig05), sig_y),
                color=color,
                s=10,
                marker="s",
                zorder=999,
            )
            print(f"% timepoints significant: {round(sum(sig05)/len(sig05)*100,2)} ({sum(sig05)}/{len(sig05)})%")

        if not skip_aesthetics:
            trial_phases = self.trial_phases if trial_phases is None else trial_phases
            plot_trial_phases(ax, trial_phases, ylim)
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

            if title is not None:
                ax.set_title(title, fontsize=18)
        return ax

    def plot_acc_indiv(
        self,
        dset=None,
        subplot_shape=None,
        significance_testing=False,
        save=False,
        title=None,
        ylim=[0, 1],
        chance_text_y=0.2,
        chance=0.5,
        skip_aesthetics=False,
        color="tab:red",
        sig_y=None,
        label=None,
        trial_phases=None,
    ):
        """
        Plots accuracy for one subject for one condition
        Arguments:
        dset (str): description of the condition to plot (can leave blank if only one condition)
        ax (matplotlib axis): axis to plot into (creates if doesn't exist)
        significance_testing (bool): run significance testing and plot significance dots
        trial_phases: dictionary or list of lists of 2 items describing when different phases occured
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

        acc, acc_shuff, t = self.dataset.get_data(dset, keys=["accuracy", "shuffledAccuracy", "times"])

        if subplot_shape is None:
            subplot_shape = (1, acc.shape[0])

        fig, axes = plt.subplots(subplot_shape[0], subplot_shape[1], figsize=(15, 5))
        axes = axes.flatten()

        for i in range(acc.shape[0]):
            acc_sub = acc[i]
            acc_shuff_sub = acc_shuff[i]

            ax = axes[i] if acc.shape[0] > 1 else axes

            acc_mean, acc_upper, acc_lower = get_plot_line(acc_sub)
            acc_shuff_mean, acc_shuff_upper, acc_shuff_lower = get_plot_line(acc_shuff_sub)

            sig_y = chance - 0.05 if sig_y is None else sig_y

            ax.plot(t, acc_mean, color=color, label=label, linewidth=2)
            ax.fill_between(t, acc_upper, acc_lower, color=color, alpha=0.5)

            ax.plot(t, acc_shuff_mean, color="gray")
            ax.fill_between(t, acc_shuff_upper, acc_shuff_lower, color="gray", alpha=0.5)

            if significance_testing:
                p, sig05 = self.do_significance_testing(t, acc_sub, acc_shuff_sub, alternative="greater")
                ax.scatter(
                    t[t > 0][sig05],
                    np.full(sum(sig05), sig_y),
                    color=color,
                    s=10,
                    marker="s",
                    zorder=999,
                )
                print(f"% timepoints significant: {round(sum(sig05)/len(sig05)*100,2)} ({sum(sig05)}/{len(sig05)})%")

            if not skip_aesthetics:
                trial_phases = self.trial_phases if trial_phases is None else trial_phases
                plot_trial_phases(ax, trial_phases, ylim)
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
                ax.set_xlabel("Time from stimulus onset (ms)")
                ax.set_ylabel("Classification accuracy", fontsize=14)
                ax.set_title(f"Sub {i}", fontsize=18)
        return fig

    def plot_hyperplane(
        self,
        labels,
        dset=None,
        ax=None,
        title=None,
        ylim=(-4, 4),
        legend_title="Trial condition",
        legend_pos="lower right",
        label_text_x=-105,
        label_text_ys=(-3.4, 2.8),
        arrow_ys=(-1.1, 1.2),
        arrow_labels=None,
        significance_testing=False,
        sig_pairs=((0, 1)),
        sig_ys=(0.2),
        alternatives=["greater"],
        sig_colors=["C0"],
        trial_phases=None,
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
        trial_phases: dictionary or list of lists of 2 items describing when different phases occured
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

        ax.plot(t, np.zeros((len(t))), "--", color="gray")

        condition_subset = [self.labels.index(label) for label in labels]
        if len(condition_subset) == 0:
            raise ValueError(f"No conditions were selected from {self.labels}")

        for i, condition in enumerate(condition_subset):
            mean, upper, lower = get_plot_line(confidence_scores[:, condition])
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

        trial_phases = self.trial_phases if trial_phases is None else trial_phases
        plot_trial_phases(ax, trial_phases, ylim)

        if significance_testing:
            sig_pairs = [self._get_pair_from_label(pair) for pair in sig_pairs]
            if len(alternatives) < len(sig_pairs):
                alternatives = alternatives + [[alternatives[0]] * (len(sig_pairs) - len(alternatives))][0]
            if len(sig_colors) < len(sig_pairs):
                sig_colors = sig_colors + [f"C{i}" for i in range(len(sig_colors), len(sig_pairs))]
            if len(sig_ys) < len(sig_pairs):
                sig_ys = sig_ys + [sig_ys[-1] + 0.2 * i for i in range(len(sig_pairs) - len(sig_ys))]

            for pair, alternative, color, y in zip(sig_pairs, alternatives, sig_colors, sig_ys):
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

        if label_text_ys is not None:
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
            plt.arrow(label_text_x, arrow_ys[0], 0, -1, head_width=45, head_length=0.25, color="k")
            plt.arrow(label_text_x, arrow_ys[1], 0, 1, head_width=45, head_length=0.25, color="k")

    def _get_pair_from_label(self, pair):
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
        title=None,
        ylim=[-1, 5],
        skip_aesthetics=False,
        color="tab:red",
        sig_y=None,
        label=None,
        trial_phases=None,
    ):
        """
        Plots accuracy for one subject for one condition
        Arguments:
        dset (str): description of the condition to plot (can leave blank if only one condition)
        ax (matplotlib axis): axis to plot into (creates if doesn't exist)
        pair: which pair to calculate
        significance_testing (bool): run significance testing and plot significance dots
        trial_phases: dictionary or list of lists of 2 items describing when different phases occured
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
        contrast = np.mean(confidence_scores[:, :, pair[1]] - confidence_scores[:, :, pair[0]], axis=1)

        mean, upper, lower = get_plot_line(contrast)

        sig_y = -0.05 if sig_y is None else sig_y

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(t, mean, color=color, label=label, linewidth=2)
        ax.fill_between(t, upper, lower, color=color, alpha=0.5)

        if significance_testing:
            p, sig05 = self.do_significance_testing(t, contrast, 0, alternative="greater")
            ax.scatter(
                t[t > 0][sig05],
                np.full(sum(sig05), sig_y),
                color=color,
                s=10,
                marker="s",
                zorder=999,
            )
            print(f"% timepoints significant: {round(sum(sig05)/len(sig05)*100,2)} ({sum(sig05)}/{len(sig05)})%")

        if not skip_aesthetics:
            trial_phases = self.trial_phases if trial_phases is None else trial_phases
            plot_trial_phases(ax, trial_phases, ylim)
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
        sig_ys=(-0.5, -0.6),
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
            sig_y=sig_ys[0],
            **kwargs,
        )
        self.plot_hyperplane_contrast(
            dset=dsets[1],
            color=colors[1],
            pair=pairs[1],
            label=labels[1],
            ax=ax,
            skip_aesthetics=True,
            sig_y=sig_ys[1],
            **kwargs,
        )
        ax.legend(custom_lines, labels, loc="lower right", frameon=True, fontsize=11)

        if significance_between:
            cs1, t = self.dataset.get_data(dsets[0], keys=["confidenceScores", "times"])
            cs2 = self.dataset.get_data(dsets[1], keys=["confidenceScores"])
            pair0 = self._get_pair_from_label(pairs[0])
            pair1 = self._get_pair_from_label(pairs[1])

            contrast_1 = np.mean(cs1[:, :, pair0[1]] - cs1[:, :, pair0[0]], axis=1)
            contrast_2 = np.mean(cs2[:, :, pair1[1]] - cs2[:, :, pair1[0]], axis=1)

            p, sig05 = self.do_significance_testing(t, contrast_1, contrast_2, alternative=test_tail)
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
        t_end=None,
        ax=None,
        significance_testing=False,
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

        t_end = t[-1] + 1 if t_end is None else t_end
        t_ix = np.logical_and(t > t_start, t < t_end)

        contrasts = [
            np.mean(cs1[:, :, pair0[1]][..., t_ix] - cs1[:, :, pair0[0]][..., t_ix], axis=(1, 2)),
            np.mean(cs2[:, :, pair1[1]][..., t_ix] - cs2[:, :, pair1[0]][..., t_ix], axis=(1, 2)),
        ]

        ax = sns.barplot(data=contrasts, errorbar="se", ax=ax, **kwargs)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.hlines(0, -0.5, 1.5, "gray", "--")
        ax.set_ylabel("Hyperplane Contrast (a.u.)")
        ax.set_xticks([0, 1], labels)

        plt.tight_layout()

        if significance_testing:
            for i, contrast in enumerate(contrasts):
                stats = pg.ttest(contrast, 0)
                print(pairs[i], stats)
                p = stats["p-val"].values[0]
                ax.text(i, sig_y + 0.05, f'p={round(p,3)}\nBF10 = {stats["BF10"].values[0]}', ha="center")

        if significance_between:

            sig_y = sig_y + 0.5 if significance_testing else sig_y
            stats = pg.ttest(contrasts[0], contrasts[1], paired=True, alternative=test_tail)
            p = stats["p-val"].values[0]
            ax.plot([0, 1], [sig_y, sig_y], "k")

            ax.set_title(title)
            ax.text(0.5, sig_y + 0.05, f'p = {round(p,3)}\nBF10 = {stats["BF10"].values[0]}', ha="center")
            print("Between:", stats)

    ## temp gen functions

    def plot_temp_gen(
        self,
        dset: str | None,
        key: str = "confidenceScores",
        pair: list | tuple | None = None,
        label: str | int | None = None,
        trial_phases: dict | None = None,
        null_metric: int | str | np.ndarray = 0,
        significance_testing: bool = False,
        test_tail: int = 0,
        cluster_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Plots a temporal generalization heatmap for a given dataset.

        Arguments:
        ----------
        dset (str or None): dataset to get data from
        key (str): identifier for which analysis to retrieve (defaults to hyperplane distances)
        pair (list or tuple or None): pair of conditions to contrast. If None, label must be provided
        label (str or int or None): single condition to plot. If None, pair must be provided
        trial_phases (dict or None): trial phases dict
        null_metric (float or ndarray): value to use as null metric for significance testing
        significance_testing (bool): whether to perform significance testing
        cluster_kwargs (dict or None): additional kwargs to pass to the cluster line collection
        **kwargs: additional kwargs to pass to sns.heatmap


        """

        tg_data, t = self.dataset.get_data(dset, keys=[key, "times"])

        if pair is None and label is not None:
            label = self._get_pair_from_label([label])[0]
            tg_data = tg_data[:, :, label]

        elif pair is not None and label is None:
            pair = self._get_pair_from_label(pair)
            tg_data = tg_data[:, :, pair[1]] - tg_data[:, :, pair[0]]

        elif pair is not None and label is not None:
            raise ValueError("Must provide either pair or label (or neither), not both")

        tg_data = tg_data.mean(1)  # average over iterations
        ax = sns.heatmap(tg_data.mean(0), **kwargs)

        trial_phases = self.trial_phases if trial_phases is None else trial_phases
        convert_t = lambda i: ((i - t[0]) / (t[-1] - t[0])) * (len(t) - 1)  # convert to place in scale

        # ticks at multiples of 250ms

        tick_multiplier = 250
        ticks = t[np.where(t % tick_multiplier == min(t % tick_multiplier))[0]]
        ax.set_xticks([convert_t(tick) for tick in ticks], labels=ticks)
        ax.set_yticks([convert_t(tick) for tick in ticks], labels=ticks)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for phase, (t0, t1) in trial_phases.items():
            ax.vlines(convert_t(t0), *ylim, colors="w", linestyles="--", linewidth=0.5)
            ax.vlines(convert_t(t1), *ylim, colors="w", linestyles="--", linewidth=0.5)
            ax.hlines(convert_t(t0), *xlim, colors="w", linestyles="--", linewidth=0.5)
            ax.hlines(convert_t(t1), *xlim, colors="w", linestyles="--", linewidth=0.5)

            ax.text(convert_t(np.mean([t0, t1])), ylim[1] - 1, phase, color="k", ha="center", va="bottom", fontsize=12)

        ax.plot((0, len(t)), (0, len(t)), "w--", linewidth=0.5)

        if significance_testing:

            if type(null_metric) == str:
                null_metric = self.dataset.get_data(dset, keys=null_metric).mean(1)

            _, clusters, cluster_p, _ = permutation_cluster_1samp_test(
                tg_data - null_metric,
                tail=test_tail,
                n_permutations=5000,
            )
            lines = []
            for i, cluster in enumerate(clusters):

                if cluster_p[i] < 0.05:
                    print(cluster_p[i])
                    mask = np.zeros(tg_data.shape[1:], dtype=bool)
                    mask[cluster] = True
                    mask = mask.T  # transpose to align with the plot
                    for px in range(130):
                        for py in range(130):
                            if mask[px, py] and not mask[px - 1, py]:
                                lines.append([[px - 0.5, py - 0.5], [px - 0.5, py + 0.5]])
                            if mask[px, py] and not mask[px + 1, py]:
                                lines.append([[px + 0.5, py - 0.5], [px + 0.5, py + 0.5]])
                            if mask[px, py] and not mask[px, py - 1]:
                                lines.append([[px - 0.5, py - 0.5], [px + 0.5, py - 0.5]])
                            if mask[px, py] and not mask[px, py + 1]:
                                lines.append([[px - 0.5, py + 0.5], [px + 0.5, py + 0.5]])

            if len(lines) > 0:
                lc = (
                    mpl.collections.LineCollection(lines, **cluster_kwargs)
                    if cluster_kwargs is not None
                    else mpl.collections.LineCollection(lines, colors="w")
                )
                ax.add_collection(lc)

        return ax, clusters, cluster_p
