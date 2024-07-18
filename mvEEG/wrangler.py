import numpy as np
import pandas as pd
import mne_bids
import mne
from pathlib import Path
from collections import defaultdict
from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split

mne.set_log_level("ERROR")


RANDOM_SEED = 42
dropped_chans_default = {
    "eeg": [],
    "eog": "ALL",
    "eyegaze": "ALL",
    "pupil": "ALL",
    "misc": "ALL",
}


class Wrangler:
    def __init__(
        self,
        data_dir,
        experiment_name,
        dropped_subs: list = [],
        dropped_chans: dict = dropped_chans_default,
        trim_timepoints=None,
        sfreq: int = 1000,
        t_win: int = 50,
        t_step: int = 25,
        trial_bin_size: int = 20,
        n_folds: int = 5,
        condition_dict: dict = None,
        conditions: list = None,
        training_groups: list = None,
        testing_groups: list = None,
    ):
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

        self.rng = np.random.default_rng(RANDOM_SEED)

        self.subs = mne_bids.get_entity_vals(
            self.bids_path.root, "subject", ignore_subjects=dropped_subs
        )
        self.nsub = len(self.subs)
        self.sfreq = sfreq
        self.trim_timepoints = trim_timepoints
        self.t_win = t_win
        self.t_step = t_step
        self.trial_bin_size = trial_bin_size
        self.n_folds = n_folds
        self.training_groups = training_groups
        self.testing_groups = testing_groups

        self.condition_dict = condition_dict
        self.conditions = conditions

        ### get general info from first subject and populate fields

        sub_path = self.bids_path.copy().update(subject=self.subs[0])
        epochs = mne.read_epochs(sub_path.update(suffix="eeg", extension=".fif").fpath)

        self.times = np.array(epochs.times * 1000).astype(int)

        if self.trim_timepoints:
            self.times = self.times[
                (self.times >= self.trim_timepoints[0])
                & (self.times <= self.trim_timepoints[1])
            ]

        t_step_ms = int(t_step // (self.times[1] - self.times[0]))
        self.t = self.times.astype(int)[t_step_ms:-t_step_ms:t_step_ms]

        self.chans_to_drop = []
        ch_names = np.array(epochs.ch_names)
        ch_types = np.array(epochs.get_channel_types())
        for chan_type in self.dropped_chans.keys():
            if self.dropped_chans[chan_type] == "ALL":
                self.chans_to_drop.extend(ch_names[ch_types == chan_type])
            else:
                self.chans_to_drop.extend(self.dropped_chans[chan_type])

        self.ch_names = np.setdiff1d(ch_names, self.chans_to_drop)

        ## make a condition dict if it doesn't exist
        if self.condition_dict is None:
            events = pd.read_csv(
                sub_path.update(suffix="events", extension=".tsv").fpath, sep="\t"
            )
            self.condition_dict = defaultdict(None)
            for trial_type in events.trial_type.unique():
                self.condition_dict[trial_type] = events[
                    events["trial_type"] == trial_type
                ].value.iloc[0]

        # if conditions is not set, generate it
        if self.conditions is None:
            self.conditions = list(self.condition_dict.keys())

        # handle grouping here
        if self.training_groups is None:
            self.training_groups = [(cond) for cond in self.conditions]
        if self.testing_groups is None:
            self.testing_groups = self.training_groups

        self.group_dict = defaultdict(None)
        for igroup, group in enumerate(self.training_groups):
            for condition in self.conditions:
                if all([subgroup in condition.split("/") for subgroup in group]):
                    self.group_dict[condition] = igroup
        for igroup, group in enumerate(self.testing_groups):
            if group not in self.training_groups:
                for condition in self.conditions:
                    if all([subgroup in condition.split("/") for subgroup in group]):
                        self.group_dict[condition] = (
                            2 + igroup
                        )  # TODO: allow for more than 2 training conditions?

        ## limit conditions to ONLY those present in group_dict

        self.condition_dict = {
            cond: self.condition_dict[cond] for cond in self.group_dict.keys()
        }
        self.conditions = list(self.group_dict.keys())

        ## calculate general information about data

        ## generate an accurate group_dict that maps

    def load_eeg(self, isub, drop_chans_manual=[], reject=True, time_bin=True):

        sub_path = self.bids_path.copy().update(subject=self.subs[isub])
        epochs = mne.read_epochs(sub_path.update(suffix="eeg", extension=".fif").fpath)
        events = pd.read_csv(
            sub_path.update(suffix="events", extension=".tsv").fpath, sep="\t"
        )["value"].to_numpy()

        # drop unwanted channels

        chans_to_drop = self.chans_to_drop.copy()
        chans_to_drop.extend(drop_chans_manual)

        epochs.drop_channels(chans_to_drop)

        if self.sfreq != epochs.info["sfreq"]:  # resample if high sampling frequency
            assert (
                epochs.info["sfreq"] % self.sfreq == 0
            ), "Cannot resample to desired frequency"
            epochs = epochs.decimate(self.sfreq)

        if self.trim_timepoints is not None:  # crop trial duration
            epochs.crop(
                tmin=self.trim_timepoints[0],
                tmax=self.trim_timepoints[1],
                include_tmax=False,
            )

        if reject:  # drop artifact marked trials
            artifacts = np.load(
                sub_path.update(suffix="rejection_flags", extension=".npy").fpath
            )

            epochs.drop(artifacts)
            events = events[~artifacts]

        xdata = epochs.get_data()

        xdata, events = self._select_labels(xdata, events)
        if time_bin:  # average within time bins (disable for ERP analyses)
            xdata_time_binned = np.zeros((xdata.shape[0], xdata.shape[1], len(self.t)))
            for tidx, t in enumerate(self.t):
                timepoints = (self.times >= t - self.t_win // 2) & (
                    self.times <= t + self.t_win // 2
                )
                xdata_time_binned[:, :, tidx] = xdata[:, :, timepoints].mean(-1)
            xdata = xdata_time_binned

        return xdata, events

    def _select_labels(self, xdata, ydata):
        """
        selects only trials with labels in self.group_dict
        - i.e., trials that will be used for decoding later
        Args:
            xdata: np.ndarray, shape (n_trials,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_trials,)

        Returns:
            xdata: np.ndarray, shape (n_trials,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_trials,)
        """

        codes = [self.condition_dict[condition] for condition in self.group_dict.keys()]
        included_trials = np.in1d(ydata, codes)

        return xdata[included_trials], ydata[included_trials]

    def _equalize_conditions(self, xdata, ydata, n_trials_per=None):
        """
        equalizes the number of trials across conditions
        Args:
            xdata: np.ndarray, shape (n_trials,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_trials,)

        Returns:
            xdata: np.ndarray, shape (n_trials,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_trials,)
        """
        if n_trials_per is None:
            n_trials_per = np.unique(ydata, return_counts=True)[
                1
            ].min()  # minimum of all trials if unset
        codes = np.unique(ydata)

        picks_per_cond = [
            self.rng.choice(np.where(ydata == code)[0], n_trials_per, replace=False)
            for code in codes
        ]
        picks = np.concatenate(picks_per_cond)

        return xdata[picks], ydata[picks]

    def bin_trials(self, xdata, ydata, permute=True):
        """
        bins trials into trial_bin_size bins
        Args:
            xdata: np.ndarray, shape (n_trials,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_trials,)

        Returns:
            xdata: np.ndarray, shape (n_bins,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_bins,)
        """

        if permute:
            perm = self.rng.permutation(xdata.shape[0])
            xdata = xdata[perm]
            ydata = ydata[perm]

        min_per_cond = np.unique(ydata, return_counts=True)[1].min()

        min_per_cond = (
            min_per_cond - min_per_cond % self.trial_bin_size
        )  # nearest multiple of trial_bin_size

        xdata, ydata = self._equalize_conditions(
            xdata, ydata, n_trials_per=min_per_cond
        )

        n_bins = xdata.shape[0] // self.trial_bin_size

        sortix = np.argsort(ydata)
        x_sort = xdata[sortix]  # sort x such that it is grouped by condition

        # next line: reshape x_sort into n_bins bins, each bin containing self.trial_bin_size trials
        # then average across each bin to get miniblocks
        # final shape is (bins_per_condition * n_conditions,channels,timepoints)
        xdata_binned = np.reshape(
            x_sort, (n_bins, self.trial_bin_size, xdata.shape[1], xdata.shape[2])
        ).mean(1)
        ydata_binned = np.repeat(np.unique(ydata), min_per_cond // self.trial_bin_size)

        return xdata_binned, ydata_binned

    def _relabel_trials(self, ydata):
        """
        groups labels into training and testing labels
        Args:
            ydata: np.ndarray, shape (n_trials,)

        Returns:
            ydata: np.ndarray, shape (n_trials,)
        """
        map_dict = {self.condition_dict[k]: v for k, v in self.group_dict.items()}
        return np.vectorize(map_dict.get)(ydata)

    def bin_and_split(self, xdata, ydata):
        """
        generator to handle trial binning and splitting into training and testing sets
        """
        for ifold in range(self.n_folds):
            xdata_binned, ydata_binned = self.bin_trials(xdata, ydata)
            ydata_binned = self._relabel_trials(ydata_binned)
            x_train, x_test, y_train, y_test = train_test_split(
                xdata_binned, ydata_binned, test_size=0.2, stratify=ydata_binned
            )
            yield x_train, x_test, y_train, y_test
