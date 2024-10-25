import numpy as np
from pathlib import Path
import mne_bids
from collections import defaultdict


class CallableDefaultDict(dict):
    """
    A dict subclass that will dynamically generate a default value for a key if it is missing
    Like defaultdict, but the default value is based on the key
    """

    def __init__(self, default_factory, *args, **kwargs):
        self.default_factory = default_factory
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        self[key] = self.default_factory(key)
        if self[key] == []:
            raise KeyError(f"Invalid key {key}")
        return self[key]


class DataLoader:
    """
    Helper class to handle loading existing data. Will default to loading everything which matches the valid analysis run

    Args:
        root_dir (str): The root directory of the dataset
        data_type (str): Data type of analysis (eg, 'Classification','RSA', or your chosen name)
        experiment_name (str): The name of the experiment
        descriptions (list or "all"): A list of descriptions (individual runs) to preload.
            Any descriptions not specified can still be lazy loaded when called
            If "all", will load all descriptions
        subs (list): A list of subjects to load. If not specified, will load all subjects with non-nan values
    """

    def __init__(
        self,
        root_dir: str,
        data_type: str,
        experiment_name: str,
        descriptions: list | None = None,
        subs: list | None = None,
    ):

        self.root_dir = root_dir

        descriptions = [] if descriptions is None else descriptions
        subs = [] if subs is None else subs

        if len(subs) == 0:  # default to all subs
            subs = [str(s.name).strip("sub-") for s in Path(root_dir).glob("sub-*")]
        self.subs = subs

        self.base_path = mne_bids.BIDSPath(
            root=root_dir,
            task=experiment_name,
            datatype=data_type,
            extension=".npy",
            check=False,
        )

        if descriptions == "all":  # if all, load every description
            descriptions = np.unique([path.description for path in self.base_path.match()]).tolist()

        self.descriptions = descriptions

        preloaded_descs = {desc: self.load_description(desc) for desc in self.descriptions}
        self._data_dict = CallableDefaultDict(self.load_description, preloaded_descs)

    def load_description(self, description: str):
        """
        Helper function that loads a single description (run) of data into the internal dictionary

        Args:
            description (str): The description (run) to load

        """
        sub_path = self.base_path.update(description=description, extension=".npy", check=False, suffix=None)

        loaded_data = defaultdict(lambda: [])
        for dset in np.unique([path.suffix for path in sub_path.match()]):  # all possible suffixes
            sub_path.update(suffix=dset)
            for path in sub_path.match():
                if path.subject in self.subs:
                    try:
                        loaded_data[dset].append(np.load(path.fpath))
                    except FileNotFoundError:
                        loaded_data[dset].append(None)

        for dset in loaded_data.keys():
            # replace subjects without a certain value with a matrix of nans
            shape = np.unique([dat.shape for dat in loaded_data[dset] if dat is not None], axis=0)
            if (len(shape) > 1) and (len(loaded_data[dset]) > 1):

                raise RuntimeError(f"Data files have inconsistent shapes: {shape}")
            loaded_data[dset] = np.stack(
                [dat if dat is not None else np.full(shape.flatten(), np.nan) for dat in loaded_data[dset]]
            )  # concatenate over subject dimension

        if "times" in loaded_data.keys():  # do we want this to be required? Could raise an error if not

            # check that our timing aligns
            if len(np.unique(loaded_data["times"][np.isfinite(loaded_data["times"]).all(axis=1)], axis=0)) > 1:
                raise ValueError("Time indices are not consistent across subjects")

            loaded_data["times"] = loaded_data["times"][0]  # only need one copy of times

        if description not in self.descriptions:  # add on to descriptions
            self.descriptions.append(description)

        return dict(loaded_data)  # return as a non-defaultdict

    def get_data(self, dset: str | None = None, keys: list | str = []):
        """
        Helper function that returns data from the internal dataset dictionary

        Args:
            dset (str): The description (run) to load. Raises an error if not specified and more than one run is present
            keys (list or str): A list of keys to load. If not specified, will load all keys

        """
        if dset is None:
            if len(self.descriptions) > 1:
                raise ValueError(f"Must specify dset if more than one run. Valid descriptions are {self.descriptions}")
            else:
                dset = self.descriptions[0]

        if len(keys) == 0:
            keys = self._data_dict[dset].keys()  # default to all

        if type(keys) is str:
            keys = [keys]  # if only 1 key

        data_to_return = []
        for key in keys:
            result = self._data_dict[dset][key]
            if len(result.shape) == 1:  # 1-D data, eg times
                data_to_return.append(result)
            else:
                result = result[np.isfinite(result).reshape(result.shape[0], -1).all(axis=1)]
                # remove subs with nans
                data_to_return.append(result)

        if len(np.unique([result.shape[0] for result in data_to_return if len(result.shape) > 1])) > 1:
            raise ValueError("Data files have inconsistent numbers of valid subjects")

        if len(data_to_return) > 1:
            return data_to_return
        else:
            return data_to_return[0]
