import pandas as pd
from torch.utils.data import Dataset

import sqlite3

from solo import construct_solo_class
from wjd_constants import *


class WeimarDB(Dataset):
    def __init__(
        self,
        config,
        partition="full",
        instrument="any",
        performer=None,
    ):

        self._parse_config(config)
        self._init_database_cursor()
        self._init_melid_list(partition, instrument, performer)

    def _parse_config(self, config):
        self.dataset_type = "weimar"
        self.data_variant = config["weimar_dataset"].get("data_variant", "raw_data")
        self.database_path = config["weimar_dataset"].get("weimardb")
        self.feature_type = config["weimar_dataset"].get("feature_type", None)
        self.labeling_type = config["weimar_dataset"].get("labeling_type", None)
        self.data_folder = config["weimar_dataset"].get(self.data_variant, None)
        self.feature_variant = config["weimar_dataset"].get(
            "feature_variant", "default"
        )

        self.feature_folder = None
        if self.feature_type == "sfnmf":
            if self.feature_variant == "demucs_frontend":
                self.feature_folder = config["sfnmf_features"].get("demucs_sfnmf_path")
            elif self.feature_variant == "default":
                self.feature_folder = config["sfnmf_features"].get("sfnmf_path", None)
        elif self.feature_type == "crepe":
            self.feature_folder = config["crepe_features"].get("crepe_path", None)

        if self.data_folder:
            self.raw_data_type = "audio"
        else:
            self.raw_data_type = None

    def _init_database_cursor(self):
        self._connect = sqlite3.connect(self.database_path)
        self._cursor = self._connect.cursor()

    def _init_melid_list(self, partition, instrument, performer):
        solo_info = self.get_solo_info()
        if partition == "train":
            solo_info = solo_info[solo_info.performer.isin(TRAIN_ARTISTS)]
        elif partition == "test-full":
            solo_info = solo_info[solo_info.performer.isin(TEST_ARTISTS)]
        elif partition == "test":
            solo_info = solo_info[solo_info.performer.isin(TEST_ARTISTS)]
            solo_info = solo_info[~solo_info.melid.isin(TEST_STOPLIST.keys())]
        elif partition == "val":
            solo_info = solo_info[solo_info.performer.isin(VAL_ARTISTS)]
        if instrument in INSTRUMENTS:
            solo_info = solo_info[solo_info.instrument == instrument]
        if performer:
            solo_info = solo_info[solo_info.performer == performer]
        self._melid_list = solo_info.melid.values

    def get_solo_info(self):
        colnames_str = ", ".join(SOLO_INFO_COLNAMES)
        query = f"SELECT {colnames_str} FROM solo_info"
        return pd.DataFrame(
            self._cursor.execute(query).fetchall(), columns=SOLO_INFO_COLNAMES
        )

    def __getitem__(self, idx):

        CustomSolo = construct_solo_class(
            dataset_type=self.dataset_type,
            feature_type=self.feature_type,
            raw_data_type=self.raw_data_type,
            labeling_type=self.labeling_type,
        )

        solo = CustomSolo(
            melid=self._melid_list[idx],
            database_filepath=self.database_path,
            audio_folder=self.data_folder,
            features_folder=self.feature_folder,
        )

        return solo

    def __len__(self):
        return len(self._melid_list)
