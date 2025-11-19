import numpy as np
import pandas as pd
import mne_bids
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from .plot_utils import plot_trial_phases, get_plot_line


dropped_chans_default = {
    "eeg": [],
    "eog": "ALL",
    "eyegaze": "ALL",
    "pupil": "ALL",
    "misc": "ALL",
}


class ERP:
    def __init__(
        self,
        data_dir,
        experiment_name,
        included_subs: list | None = None,
        dropped_subs: list | None = None,
        dropped_chans: dict = dropped_chans_default,
        trial_phases: dict | None = None,
        reject: bool = True,
        colors: list = [f"C{i}" for i in range(10)],
    ):

        self.trial_phases = trial_phases
        self.colors = colors

        self.data_dir = data_dir
        self.experiment_name = experiment_name
        self.dropped_chans = dropped_chans

        self.bids_path = mne_bids.BIDSPath(
            root=self.data_dir,
            task=self.experiment_name,
            datatype="eeg",
            description="preprocessed",
            check=False,
        )

        dropped_subs = [] if dropped_subs is None else dropped_subs
        if included_subs is None:  # default to all subs
            self.subs = mne_bids.get_entity_vals(self.bids_path.root, "subject", ignore_subjects=dropped_subs)
        else:
            self.subs = included_subs
            if any([sub in included_subs for sub in dropped_subs]):
                raise ValueError(
                    "Included and dropped subjects overlap"
                    + f"subjects: {[sub for sub in dropped_subs if sub in included_subs]}"
                )

        self.nsub = len(self.subs)

        self.xdata_all = np.empty((self.nsub), dtype="object")
        self.ydata_all = np.empty((self.nsub), dtype="object")
        for isub, sub in enumerate(self.subs):
            sub_path = self.bids_path.copy().update(subject=sub)
            epochs = mne.read_epochs(sub_path.update(suffix="eeg", extension=".fif").fpath)
            events = pd.read_csv(sub_path.update(suffix="events", extension=".tsv").fpath, sep="\t")

            # drop unwanted channels
            chans_to_drop = []
            ch_names = np.array(epochs.ch_names)
            ch_types = np.array(epochs.get_channel_types())
            for chan_type in self.dropped_chans.keys():  # drop irrelevant channels
                if self.dropped_chans[chan_type] == "ALL":
                    chans_to_drop.extend(ch_names[ch_types == chan_type])
                else:
                    chans_to_drop.extend(self.dropped_chans[chan_type])
            epochs.drop_channels(chans_to_drop)

            if isub == 0:  # generate base values
                self.times = np.array(epochs.times * 1000).astype(int)
                self.condition_dict = {
                    trial_type: events[events["trial_type"] == trial_type].value.iloc[0]
                    for trial_type in events.trial_type.unique()
                }
                self.conditions = list(self.condition_dict.keys())
                self.ch_names = epochs.ch_names

            events = events["value"].to_numpy()

            if reject:  # drop artifact marked trials
                if not hasattr(self, "artifacts"):
                    self.artifacts = {}
                self.artifacts[isub] = np.load(sub_path.update(suffix="rejection_flags", extension=".npy").fpath)
                epochs.drop(self.artifacts[isub])
                events = events[~self.artifacts[isub]]

            self.xdata_all[isub] = epochs.get_data()
            self.ydata_all[isub] = events

    def _select_conditions(self, xdata, ydata, group=None):
        """
        Helper function that selects only trials with labels in 'labels'
        - i.e., trials that will be used for decoding later
        Args:
            xdata: np.ndarray, shape (n_trials,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_trials,)
            labels: list of labels to include (defaults to all)

        Returns:
            xdata: np.ndarray, shape (n_trials,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_trials,)
        """

        if group is None:
            return xdata, ydata

        if "/" in group:
            group = group.split("/")
        elif isinstance(group, str):
            group = [group]

        labels = [cond for cond in self.conditions if all([subgroup in cond.split("/") for subgroup in group])]
        codes = [self.condition_dict[condition] for condition in labels]
        included_trials = np.in1d(ydata, codes)

        return xdata[included_trials], ydata[included_trials]

    def _select_electrodes(self, xdata, subset: str | list = None):

        if subset is None:
            return xdata

        if isinstance(subset, str):

            el_ix = np.in1d(self.ch_names, [ch for ch in self.ch_names if ch.startswith(subset)])

        elif len(subset) > 1:
            chans = np.concatenate([[ch for ch in self.ch_names if ch.startswith(ss)] for ss in subset])
            el_ix = np.in1d(self.ch_names, chans)
        else:
            raise ValueError("subset must be a string or list of strings")

        return xdata[:, el_ix, :]

    def plot_erp(
        self, condition_subsets, electrode_subsets, ax=None, labels=None, subs=None, trial_phases=None, ylim=None
    ):

        if ax is None:
            _, ax = plt.subplots()

        subs = np.arange(self.nsub) if subs is None else subs
        nsub = len(subs)

        # equate lengths if unequal
        if len(condition_subsets) == 1 and len(electrode_subsets) > 1:
            condition_subsets = condition_subsets * len(electrode_subsets)
            leg_labels = electrode_subsets

        elif len(electrode_subsets) == 1 and len(condition_subsets) > 1:
            electrode_subsets = electrode_subsets * len(condition_subsets)
            leg_labels = condition_subsets

        elif len(condition_subsets) == len(electrode_subsets):
            leg_labels = [f"({cond},{el})" for cond, el in zip(condition_subsets, electrode_subsets)]
        else:
            raise ValueError(
                "condition_subsets and electrode_subsets must be of equal length or one must be of length 1"
            )

        labels = leg_labels if labels is None else labels

        for i, (conds, els) in enumerate(zip(condition_subsets, electrode_subsets)):

            sub_erp = np.empty((nsub, len(self.times)))
            for isub in range(nsub):
                xdata, _ = self._select_conditions(self.xdata_all[subs[isub]], self.ydata_all[subs[isub]], group=conds)
                xdata = self._select_electrodes(xdata, els)
                if len(xdata.shape) == 2:
                    sub_erp = xdata.mean(1)  # average across trials
                else:
                    sub_erp[isub] = xdata.mean((0, 1))  # average across times

            mean, upper, lower = get_plot_line(sub_erp)
            ax.plot(self.times, mean, label=labels[i], color=self.colors[i])
            ax.fill_between(self.times, upper, lower, alpha=0.2, color=self.colors[i])
        plt.legend()

        ## Aesthetics
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(r"ERP amplitude ($\mu$V)")
        ax.invert_yaxis()
        trial_phases = self.trial_phases if trial_phases is None else trial_phases
        plot_trial_phases(ax, trial_phases, ax.get_ylim())
