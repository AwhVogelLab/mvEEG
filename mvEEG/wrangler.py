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
    """
    Class to handle data loading, binning, and handling

    Args:
        data_dir (str): The root directory of the dataset (eg experiment/derivatives)
        experiment_name (str): The name of the experiment
        dropped_subs (list): A list of subjects to drop (for example, too many artifacts, broken data)
        dropped_chans (dict): A dictionary of which channels to drop.
            Default: no EEG channels, ALL other channels (eog, eyegaze, pupil, misc)
        trim_timepoints (tuple): A tuple of the timepoints to trim the data to (start, end) - defaults to none
        sfreq (int), default 1000: The sampling frequency of the data in Hz
        t_win (int), default 50: The window size to bin trials in ms 
        t_step (int), default 25: window step size
            If t_step < t_win, will be a sliding window
        trial_bin_size (int), default 20: The number of trials to bin together - Set to 1 for no binning
        n_folds (int): The number of crossvalidation folds passed to train_test_split - default 100, preferred values are 1000 for decoding or 10000 for RSA
        condition_dict (dict): A dictionary mapping trial types to event codes
            If not specified, will use all labels and event codes found in events.tsv
        conditions (list): A list of conditions to use for decoding. Defaults to all conditions in condition_dict
            If you want to select conditions by sub-categories, they should be formatted in MNE style:
                label1/label2,label3/label4, eg color/ss1/left
        training_groups (list): A list of conditions to use for training. Defaults to all conditions
        testing_groups (list): A list of conditions to use for testing. Defaults to training_groups
            If set, give these as a list of conditions in the format [label1/label2,label3/label4]
            or, alternatively [(label1,label2),(label3,label4)]. Both of these will group conditions with ALL of the specified labels in their names
            Alternatively, just give a list of condition names
    """

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
        n_folds: int = 100,
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

        self.subs = mne_bids.get_entity_vals(self.bids_path.root, "subject", ignore_subjects=dropped_subs)
        self.nsub = len(self.subs)
        self.sfreq = sfreq
        self.trim_timepoints = trim_timepoints
        self.t_win = t_win
        self.t_step = t_step
        self.trial_bin_size = trial_bin_size
        self.n_folds = n_folds
        self.condition_dict = condition_dict
        self.conditions = conditions

        ### get general info from first subject and populate fields

        sub_path = self.bids_path.copy().update(subject=self.subs[0])
        epochs = mne.read_epochs(sub_path.update(suffix="eeg", extension=".fif").fpath)

        self.times = np.array(epochs.times * 1000).astype(int)

        if self.trim_timepoints:
            self.times = self.times[(self.times >= self.trim_timepoints[0]) & (self.times <= self.trim_timepoints[1])]

        # get timepoints to start our bins at
        t_step_ms = int(t_step // (self.times[1] - self.times[0]))
        self.t = self.times.astype(int)[t_step_ms:-t_step_ms:t_step_ms]

        self.chans_to_drop = []
        ch_names = np.array(epochs.ch_names)
        ch_types = np.array(epochs.get_channel_types())
        for chan_type in self.dropped_chans.keys(): # drop irrelevant channels
            if self.dropped_chans[chan_type] == "ALL":
                self.chans_to_drop.extend(ch_names[ch_types == chan_type])
            else:
                self.chans_to_drop.extend(self.dropped_chans[chan_type])

        self.ch_names = np.setdiff1d(ch_names, self.chans_to_drop)

        ## make a condition dict if it doesn't exist
        if self.condition_dict is None:
            events = pd.read_csv(sub_path.update(suffix="events", extension=".tsv").fpath, sep="\t")
            self.condition_dict = defaultdict(None)
            for trial_type in events.trial_type.unique():
                self.condition_dict[trial_type] = events[events["trial_type"] == trial_type].value.iloc[0]

        # if conditions is not set, generate it
        if self.conditions is None:
            self.conditions = list(self.condition_dict.keys())

        # handle grouping here
        if training_groups is None:
            training_groups = [(cond) for cond in self.conditions]
        if testing_groups is None:
            testing_groups = training_groups

        training_groups = [cond.split("/") if "/" in cond else cond for cond in training_groups]  # split into subgroups
        testing_groups = [cond.split("/") if "/" in cond else cond for cond in testing_groups]

        self.training_conditions = []
        self.testing_conditions = []


        ## auto generate a grouping dict based on training and testing conditions.
        ## This translates EEG condition codes into ordered codes (for regression analyses)
        ## TODO: make this exposed to the user?



        self.group_dict = defaultdict(None)

        for igroup, group in enumerate(training_groups):
            for condition in self.conditions:
                if all([subgroup in condition.split("/") for subgroup in group]): # check if all subgroups are in condition
                    self.group_dict[condition] = igroup # assign labels starting at 0
                    self.training_conditions.append(condition)
        for igroup, group in enumerate(testing_groups):
            if group not in training_groups:
                for condition in self.conditions:
                    if all([subgroup in condition.split("/") for subgroup in group]):
                        self.group_dict[condition] = (
                            len(training_groups) + igroup
                        )  # offset by number of training groups
                        self.testing_conditions.append(condition)
            else:
                for condition in self.conditions:
                    if all([subgroup in condition.split("/") for subgroup in group]):
                        self.testing_conditions.append(condition)

    def load_eeg(self, isub, drop_chans_manual=[], reject=True, time_bin=True):

        """
        Function to load in EEG data for a given subject and do basic preprocessing

        Args:
            isub (int): index of the subject to load
            drop_chans_manual (list): list of channels to drop manually for this subject
            reject (bool), default True: whether to drop trials marked as artifacts
            time_bin (bool), default True: whether to time bin the data. Disable for ERP analyses. Must be enabled for decoding/rsa

        Returns:
            xdata: np.ndarray of EEG data, shape (n_trials,n_chans,n_timepoints)
            ydata: np.ndarray of selected, shape (n_trials,)
        """

        # load in epoched EEG data and events

        sub_path = self.bids_path.copy().update(subject=self.subs[isub])
        epochs = mne.read_epochs(sub_path.update(suffix="eeg", extension=".fif").fpath)
        events = pd.read_csv(sub_path.update(suffix="events", extension=".tsv").fpath, sep="\t")["value"].to_numpy()

        # drop unwanted channels

        chans_to_drop = self.chans_to_drop.copy()
        chans_to_drop.extend(drop_chans_manual)

        chans_to_drop = [chan for chan in chans_to_drop if chan in epochs.ch_names]
        epochs.drop_channels(chans_to_drop)

        if self.sfreq != epochs.info["sfreq"]:  # check if resampling options are valid. 
            if epochs.info["sfreq"] % self.sfreq != 0:
                raise ValueError(f"Cannot resample EEG data to target frequency (not an integer multiple). 
                                 Data sampling rate is {epochs.info['sfreq']} Hz,
                                 but the requested sampling rate is {self.sfreq} Hz.")
            if epochs.info["sfreq"] < self.sfreq:
                raise ValueError(f"Cannot upsample EEG data. 
                                 Data sampling rate is {epochs.info['sfreq']} Hz,
                                 but the requested sampling rate is {self.sfreq} Hz.")

            epochs = epochs.decimate(self.sfreq / epochs.info["sfreq"]) # resample by decimating

        if self.trim_timepoints is not None:  # crop trial duration
            epochs.crop(
                tmin=self.trim_timepoints[0],
                tmax=self.trim_timepoints[1],
                include_tmax=False,
            )

        if reject:  # drop artifact marked trials
            artifacts = np.load(sub_path.update(suffix="rejection_flags", extension=".npy").fpath)

            epochs.drop(artifacts)
            events = events[~artifacts]

        xdata = epochs.get_data()

        xdata, events = self._select_labels(xdata, events) # select out labels in conditions
        if time_bin:  # average within time bins (disable for ERP analyses)
            xdata_time_binned = np.zeros((xdata.shape[0], xdata.shape[1], len(self.t)))
            for tidx, t in enumerate(self.t): # average over time dimension
                timepoints = (self.times >= t - self.t_win // 2) & (self.times <= t + self.t_win // 2)
                xdata_time_binned[:, :, tidx] = xdata[:, :, timepoints].mean(-1)
            xdata = xdata_time_binned

        return xdata, events

    def _select_labels(self, xdata, ydata, labels=None, code_dict=None):
        """
        Helper function that selects only trials with labels in 'labels'
        - i.e., trials that will be used for decoding later
        Args:
            xdata: np.ndarray, shape (n_trials,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_trials,)
            labels: list of labels to include (defaults to all included in self.group_dict)
            code_dict: dict, mapping labels to codes (defaults to self.condition_dict)
                - alternatively set to self.group_dict to use group_dict for labels

        Returns:
            xdata: np.ndarray, shape (n_trials,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_trials,)
        """
        if labels is None:
            labels = self.group_dict.keys()
        if code_dict is None:
            code_dict = self.condition_dict

        codes = [code_dict[condition] for condition in labels]
        included_trials = np.in1d(ydata, codes)

        return xdata[included_trials], ydata[included_trials]

    def _equalize_conditions(self, xdata, ydata, n_trials_per=None):
        """
        equalizes the number of trials across conditions
        Args:
            xdata: np.ndarray, shape (n_trials,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_trials,)
            n_trials_per: int, number of trials to include per condition
                defaults to minimum number of condition albels

        Returns:
            xdata: np.ndarray, shape (n_trials,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_trials,)
        """

        if n_trials_per is None:
            n_trials_per = np.unique(ydata, return_counts=True)[1].min()  # minimum of all trials if unset
        codes = np.unique(ydata)

        # randomly pick n_trials_per trials from each condition
        picks_per_cond = [self.rng.choice(np.where(ydata == code)[0], n_trials_per, replace=False) for code in codes]
        picks = np.concatenate(picks_per_cond)

        return xdata[picks], ydata[picks]

    def bin_trials(self, xdata, ydata, permute=True):
        """
        bins trials into trial_bin_size bins
        Args:
            xdata: np.ndarray, shape (n_trials,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_trials,)
            permute: bool, default True, whether to permute the data before binning.

        Returns:
            xdata: np.ndarray, shape (n_bins,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_bins,)
        """

        if permute: # shuffle trials if set
            perm = self.rng.permutation(xdata.shape[0])
            xdata = xdata[perm]
            ydata = ydata[perm]

        min_per_cond = np.unique(ydata, return_counts=True)[1].min()

        min_per_cond = min_per_cond - min_per_cond % self.trial_bin_size  # nearest multiple of trial_bin_size

        xdata, ydata = self._equalize_conditions(xdata, ydata, n_trials_per=min_per_cond) # equalize trials across conditions

        if self.trial_bin_size == 1:  # if not binning, just return the trimmed inputs
            return xdata, ydata

        n_bins = xdata.shape[0] // self.trial_bin_size

        # sort x such that it is grouped by condition
        # this lets us bin by reshaping the data instead of looping over trials
        sortix = np.argsort(ydata)
        x_sort = xdata[sortix] 
        

        # next line: reshape x_sort into n_bins bins, each bin containing self.trial_bin_size trials
        # so (trials x channels x timepoints) -> (bins x trials_per_bin x channels x timepoints)
        # then average across each bin to get miniblocks
        # final shape is (bins_per_condition * n_conditions,channels,timepoints)
        xdata_binned = np.reshape(x_sort, (n_bins, self.trial_bin_size, xdata.shape[1], xdata.shape[2])).mean(1)
        # ydata is just the equivalent bin labels
        ydata_binned = np.repeat(np.unique(ydata), min_per_cond // self.trial_bin_size)

        return xdata_binned, ydata_binned

    def _relabel_trials(self, ydata,og_dict = None, new_dict = None):
        """
        
        Helper function to relabel trials from one set of codes to another.
        Used to transform event codes to group codes for decoding.

        Conditions are renamed from those in og_dict (default condition_dict) to those in new_dict (default group_dict)

        Args:
            ydata: np.ndarray, shape (n_trials,)
            og_dict (dict), defaut self.condition_dict: dict mapping trial labels to codes
            new_dict (dict), default self.group_dict: dict mapping trial labels to new codes

        Returns:
            ydata: np.ndarray, shape (n_trials,)
        """

        if og_dict is None:
            og_dict = self.condition_dict
        if new_dict is None:
            new_dict = self.group_dict

        map_dict = {og_dict[k]: v for k, v in new_dict.items()}
        return np.vectorize(map_dict.get)(ydata)

    def bin_and_split(self, xdata, ydata, test_size=0.2):
        """
        generator to handle trial binning and splitting into training and testing sets.

        Args:
            xdata: np.ndarray, shape (n_trials,n_chans,n_timepoints)
            ydata: np.ndarray, shape (n_trials,)
            test_size: float (default 0.2), proportion of data to use for testing
        
        Returns:
            x_train: np.ndarray, shape (n_train,n_chans,n_timepoints) -  EEG training data
            x_test: np.ndarray, shape (n_test,n_chans,n_timepoints) - EEG testing data
            y_train: np.ndarray, shape (n_train,) -  training labels
            y_test: np.ndarray, shape (n_test,) -  testing labels

        """

        for self.ifold in range(self.n_folds):
            xdata_binned, ydata_binned = self.bin_trials(xdata, ydata) # bin trials and relabel conditions
            ydata_binned = self._relabel_trials(ydata_binned)


            x_train, x_test, y_train, y_test = train_test_split(
                xdata_binned, ydata_binned, test_size=test_size, stratify=ydata_binned
            ) # do main train_test split. Stratifies such that there are an equal proportion of each condition in each set


            # if cross-decoding ensure that appropriate conditions appear in each set
            if self.training_conditions != self.testing_conditions: 

                x_train, y_train = self._select_labels(
                    x_train, y_train, self.training_conditions, code_dict=self.group_dict
                )
                x_test, y_test = self._select_labels(x_test, y_test, self.testing_conditions, code_dict=self.group_dict)

            yield x_train, x_test, y_train, y_test
