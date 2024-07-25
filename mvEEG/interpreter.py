import numpy as np
import mne_bids
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sista
import statsmodels as sm
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from collections import defaultdict

mpl.rcParams["font.sans-serif"] = "Arial"
mpl.rcParams["font.family"] = "sans-serif"
import os
import re
import pingouin as pg


class Interpreter:
    def __init__(
        self,
        labels: list,
        data_dir: str,
        experiment_name: str,
        descriptions: list = [],
        subs: list = [],
    ):
        self.labels = labels
        if len(subs) == 0:  # default to all subs
            subs = [str(s.name).strip("sub-") for s in Path(data_dir).glob("sub-*")]
        self.subs = subs

        if len(descriptions) == 0:  # default to every possible description present
            descriptions = np.unique(
                np.concatenate(
                    [
                        np.unique(
                            [
                                re.findall("desc-(.*)_", s.name)
                                for s in Path(os.path.join(data_dir, f"sub-{sub}","classification")).glob(
                                    "*.npy"
                                )
                            ]
                        )
                        for sub in self.subs
                    ]
                )
            )
        self.descriptions = descriptions

        self.data_dict = {}

        for description in self.descriptions:  # load in data

            sub_path = mne_bids.BIDSPath(
                root=data_dir,
                task=experiment_name,
                datatype="classification",
                description=description,
                extension=".npy",
                check=False,
            )

            loaded_data = defaultdict(lambda: [])

            for sub in subs:  # load each subject's data from disc
                sub_path.update(subject=sub)
                for dset in [
                    "accuracy",
                    "shuffledAccuracy",
                    "confusionMatrix",
                    "confidenceScores",
                    "times",
                ]:
                    sub_path.update(suffix=dset)
                    try:
                        loaded_data[dset].append(np.load(sub_path.fpath))
                    except FileNotFoundError:
                        loaded_data[dset].append(None)

            for dset in loaded_data.keys():
                 # replace subjects without a certain value with a matrix of nans
                shape = np.unique(
                    [dat.shape for dat in loaded_data[dset] if dat is not None], axis=0
                )
                if (len(shape) > 1) and (len(loaded_data[dset]) > 1):

                    raise RuntimeError(f"Data files have inconsistent shapes: {shape}")
                loaded_data[dset] = [
                    dat if dat is not None else np.full(shape.flatten(), np.nan)
                    for dat in loaded_data[dset]
                ]

            self.data_dict.update(
                {description: {k: np.stack(v) for k, v in loaded_data.items()}}
            )

            # check that our timing aligns
            if len(np.unique(self.data_dict[description]["times"][np.isfinite(self.data_dict[description]["times"]).all(axis=1)], axis=0)) > 1:
                raise ValueError("Time indices are not consistent across subjects")
            self.data_dict[description]["times"] = self.data_dict[description]["times"][
                0
            ]

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
        Arguments:
        ax: axis to plot into
        stim_time: iterable of 2 ints (for a single stim bar), OR iterable of iterables of 2 ints (multiple bars)
        ylim: y limits of figure
        hide (bool): set to true to not actually plot the bar but return an aggregate
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
        Helper function that runs significance testing at each timepoint and determines the appropriate test

        Arguments:
        t: 1D array of time points
        a: 2D array of shape [subjects,time points]
        b: 2D array of shape [subjects,time points] or scalar (for 1 sample test)
        test: test to run, defaults to ttest_rel if b is 2D, ttest_1samp if b is scalar
        alternative: 'two-sided' or 'greater' or 'less'
        correction_method: method to correct for multiple comparisons, defaults to 'fdr_bh'


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

    def _get_data(self, dset=None, keys=[]):
        """
        Helper function that returns data from the internal dataset dictionary

        """
        if dset is None:
            if len(self.descriptions) > 1:
                raise ValueError(
                    f"Must specify dset if more than one run. Valid descriptions are {self.descriptions}"
                )
            else:
                dset = self.descriptions[0]

        if len(keys) == 0:
            keys = self.data_dict[dset].keys()


        data_to_return = []
        for key in keys:
            result = self.data_dict[dset][key]
            if len(result.shape) == 1: # 1-D data, eg times
                data_to_return.append(result)
            else:
                result = result[np.isfinite(result).reshape(len(self.subs),-1).all(axis=1)]
                # remove subs with nans
                data_to_return.append(result)

        if len(np.unique(result.shape for result in data_to_return)) > 1:
            raise ValueError("Data files have inconsistent numbers of valid subjects")



        if len(data_to_return) > 1:
            return data_to_return
        else:
            return data_to_return[0]

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

        acc, acc_shuff, t = self._get_data(
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
            ax.legend(loc="lower right", frameon=False, fontsize=11)

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

        confidence_scores, t = self._get_data(dset, keys=["confidenceScores", "times"])

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

        confidence_scores, t = self._get_data(dset, keys=["confidenceScores", "times"])
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

        Input:
        self.conf_mat of shape [subjects,timepoints,folds,setsizeA,setsizeB]
        """

        conf_mat, t = self._get_data(dset, keys=["confusionMatrix", "times"])

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
        **kwargs,
    ):

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
            **kwargs,
        )
        self.plot_hyperplane_contrast(
            dset=dsets[1],
            color=colors[1],
            pair=pairs[1],
            label=labels[1],
            ax=ax,
            skip_aesthetics=True,
            sig_y=-0.1,
            **kwargs,
        )
        ax.legend(custom_lines, labels, loc="lower right", frameon=True, fontsize=11)

        if significance_between:
            cs1, t = self._get_data(dsets[0], keys=["confidenceScores", "times"])
            cs2 = self._get_data(dsets[1], keys=["confidenceScores"])
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
        colors=["C0", "C1"],
        ax=None,
        significance_between=False,
        sig_y=1,
        sig_color="C2",
        test_tail="two-sided",
        ylim=None,
        title=None,
        **kwargs,
    ):

        if ax is None:
            _, ax = plt.subplots()

        cs1, t = self._get_data(dsets[0], keys=["confidenceScores", "times"])
        cs2 = self._get_data(dsets[1], keys=["confidenceScores"])
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
        ax.set_xticks([0, 1], labels)

        stats = pg.ttest(contrasts[0], contrasts[1], paired=True, alternative=test_tail)
        p = stats["p-val"].values[0]

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.plot([0, 1], [sig_y, sig_y], "k")
        ax.hlines(0, -0.5, 1.5, "gray", "--")
        ax.set_ylabel("Hyperplane Contrast (a.u.)")
        
        
        plt.tight_layout()

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