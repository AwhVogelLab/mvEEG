import numpy as np
from pathlib import Path
import mne_bids
import os
import re
from collections import defaultdict

class DataLoader:
    def __init__(
        self,
        root_dir: str,
        data_type: str,
        experiment_name: str,
        descriptions: list = [],
        subs: list = []
    ):

        self.root_dir = root_dir
        if len(subs) == 0:  # default to all subs
            subs = [str(s.name).strip("sub-") for s in Path(root_dir).glob("sub-*")]
        self.subs = subs

        base_path = mne_bids.BIDSPath(
            root=root_dir,
            task=experiment_name,
            datatype=data_type,
            extension='.npy',
            check=False)

        if len(descriptions) == 0:  # default to every possible description present
            descriptions = np.unique([path.description for path in base_path.match()]).tolist()


        self.descriptions = descriptions

        self.data_dict = {}
        

        for description in self.descriptions:  # load in data

            sub_path = base_path.update(
                description=description,
                extension=".npy",
                check=False
            )

            loaded_data = defaultdict(lambda: [])

            for dset in np.unique([path.suffix for path in sub_path.match()]): # all possible suffixes
                sub_path.update(suffix=dset)
                for path in sub_path.match():
                    try:
                        loaded_data[dset].append(np.load(path.fpath))
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
            self.data_dict[description]["times"] = self.data_dict[description]["times"][0]


    def get_data(self, dset=None, keys=[]):
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