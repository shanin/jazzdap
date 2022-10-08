import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchaudio
from math import floor


class WeimarDB(Dataset):
    def __init__(self, config):
        self.index = pd.read_csv(
            os.path.join(
                config['metadata_path'], 
                config['youtube_index']
            )
        )
        self.audio_dir = os.path.join(
            config['data_path'], 
            config['audio_dir']
        )
        self.filenames = os.listdir(self.audio_dir)
        self.config = config
    
    def __len__(self):
        raise NotImplemented

    def _get_metadata(self, idx):
        raise NotImplemented

    def __getitem__(self, idx):
        meta = self._get_metadata(idx)
        data, sample_rate = torchaudio.load(
            os.path.join(
                self.audio_dir, 
                f'{meta.youtube_id}.mp3'
            )
        )
        start = floor(meta.solo_start_sec * sample_rate)
        stop = floor(meta.solo_end_sec * sample_rate)
        solo = data[:, start:stop]
        sample = {
            'audio': solo,
            'sample_rate': sample_rate,
            'query': meta['query']
        }
        return sample

class WeimarDBfull(WeimarDB):
    def __len__(self):
        return self.index.shape[0]

    def _get_metadata(self, idx):
        return self.index.iloc[idx]

class WeimarDBunique(WeimarDB):
    def __init__(self, config):
        super().__init__(config)
        self.unique_index = self.index.drop_duplicates(subset=['query'])
    
    def __len__(self):
        return self.unique_index.shape[0]

    def _get_metadata(self, idx):
        return self.unique_index.iloc[idx]