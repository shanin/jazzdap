import os
import sys

sys.path.insert(0, os.path.join(sys.path[0], '../packages/baseline_1/predict'))

import argparse
import parsing

import numpy as np
import torchaudio

from utils import load_config, safe_mkdir
from source_filter_model import main as sfm
from dataset import WeimarDB

def precompute_HF0(config):
    output_path = os.path.join(
        config['shared']['exp_folder'],
        config['sfnmf']['output_folder']
    )
    if config['sfnmf']['save']:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
    
    wdb = WeimarDB(config)
    filename = '.resampled-sf-nmf-input.wav'
    for sample in wdb:
        torchaudio.save(filename, sample.resampled_audio(), sample.resample_rate)

        Fs = config['sfnmf']['Fs']
        hop = config['sfnmf']['hop'] / Fs
        audio_fpath = filename
        input_args = [u'{0}'.format(audio_fpath), 
                      u'--samplingRate={0}'.format(Fs), 
                      u'--hopsize={0}'.format(hop)]
        
        (pargs, options) = parsing.parseOptions(input_args)

        times, HF0, HGAMMA, HPHI, WM, HM, pitch_accuracy, options = sfm(pargs, options)
        pitch_accuracy = np.array(pitch_accuracy)
        sfm_output = {
            'times': times,
            'HF0': HF0,
            'HGAMMA': HGAMMA,
            'HPHI': HPHI,
            'WM': WM,
            'HM': HM,
            'pitch_accuracy': pitch_accuracy,
            'options': options
        }
        if config['sfnmf']['save']:
            np.save(
                os.path.join(output_path, "melid_" + str(sample.melid).zfill(3)), 
                sfm_output['HF0']
            )
    os.remove(filename)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logging-level", type=str, default="WARNING")
    args = parser.parse_args()

    config = load_config(args.config)
    precompute_HF0(config)
