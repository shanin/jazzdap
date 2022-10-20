import argparse
from configparser import ConfigParser
import os

import pandas as pd
import torchaudio
from math import floor

from utils import load_config, safe_mkdir
from dataset import WeimarDB

def get_metadata(config_shared, config_local):
    file_path = os.path.join(
        config_shared['raw_data'], 
        config_local['youtube_index']
    )
    return pd.read_csv(file_path)

def filter_metadata(config_shared, config_local):
    
    audio_path = os.path.join(
        config_shared['raw_data'], 
        config_local['audio_dir']
    )
    output_dir = os.path.join(
        config_shared['exp_folder'],
        config_local['output_dir']
    )
    filtered_index_file = os.path.join(
        output_dir,
        config_local['output_file']
    )

    metadata = get_metadata(config_shared, config_local)
    file_list = os.listdir(audio_path)
    rows = []
    for _, row in metadata.iterrows():
        if f'{row.youtube_id}.mp3' in file_list:
            rows.append(row)
    filtered = pd.DataFrame(rows)
    safe_mkdir(output_dir)
    filtered.to_csv(filtered_index_file)

def save_wavs_for_baseline_evaluation(complete_config):
    config = complete_config['preprocess']
    wdb = WeimarDB(complete_config)
    wav_dir = os.path.join(
        complete_config['shared']['exp_folder'],
        config['wav_dir']
    )
    safe_mkdir(wav_dir)
    for idx in range(len(wdb)):
        sample = wdb[idx]
        query = sample['query']
        torchaudio.save(
            f'{wav_dir}/{query}.wav', 
            sample['audio'], 
            sample['sample_rate']
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logging-level", type=str, default="WARNING")
    args = parser.parse_args()
    
    config = load_config(args.config)
    #filter_metadata(config['shared'], config['preprocess'])
    save_wavs_for_baseline_evaluation(config)
