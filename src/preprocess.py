import argparse
import os

import pandas as pd
from IPython.display import Audio, display
import torchaudio
from math import floor

from utils import load_config

def get_metadata(config):
    return pd.read_csv(os.path.join(config['data_path'], config['youtube_index']))

def filter_metadata(config):
    metadata = get_metadata(config)
    file_list = os.listdir(os.path.join(config['data_path'], config['audio_dir']))
    rows = []
    for _, row in metadata.iterrows():
        if f'{row.youtube_id}.mp3' in file_list:
            rows.append(row)
    filtered = pd.DataFrame(rows)
    if not os.path.exists(config['output_dir']):
        os.mkdir(config['output_dir'])
    filtered.to_csv(
        os.path.join(
            config['output_dir'],
            config['output_file']
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logging-level", type=str, default="WARNING")
    args = parser.parse_args()
    
    config = load_config(args.config)
    filter_metadata(config['preprocess'])
