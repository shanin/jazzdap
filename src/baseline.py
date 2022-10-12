import os
import sys

sys.path.insert(0, os.path.join(sys.path[0], '../packages/baseline_1/predict'))

import argparse
import h5py
import parsing

import numpy as np

from utils import load_config
from source_filter_model import main as sfm
from predict_on_single_audio_CRNN import load_model, get_prediction, save_output

def save_HF0(hf0_dict, config, trackname):
    if not os.path.exists(config['output_folder']):
        os.mkdir(config['output_folder'])
    output_path = os.path.join(config['output_folder'], config['h5py_folder'])
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    filename = '{0}/{1}.h5'.format(output_path, trackname.split('.wav')[0])
    out = h5py.File(filename, 'w')
    out.create_dataset(
        'HF0', 
        hf0_dict['HF0'].shape, 
        data=hf0_dict['HF0']
    )
    out.create_dataset(
        'pitch_accuracy', 
        hf0_dict['pitch_accuracy'].shape, 
        data=hf0_dict['pitch_accuracy']
    )
    out.create_dataset(
        'HGAMMA', 
        hf0_dict['HGAMMA'].shape, 
        data=hf0_dict['HGAMMA']
    )
    out.create_dataset(
        'HPHI', 
        hf0_dict['HPHI'].shape, 
        data=hf0_dict['HPHI']
    )
    out.create_dataset(
        'WM', 
        hf0_dict['WM'].shape, 
        data=hf0_dict['WM']
    )
    out.create_dataset(
        'HM', 
        hf0_dict['HM'].shape, 
        data=hf0_dict['HM'])
    out.close()
    ### add parameters file??

#go to packages/baseline_1/predict/extract_HF0 for details
def extract_HF0(config, filename):
    Fs = config['Fs']
    hop = config['hop'] / Fs
    pitch_corrected = config['pitch_corrected']

    #train_parameter = 'HF0_standard'
    
    track_name_original = filename.split('.wav')[0]
    audio_fpath = '{0}/{1}'.format(config['input_data'], filename)
    print('{0} - Processing'.format(track_name_original))
    input_args = [u'{0}'.format(audio_fpath), \
                  u'--samplingRate={0}'.format(Fs), \
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
    return sfm_output


def run_baseline_single_call(config, filename):
    hf0_dict = extract_HF0(config, filename)
    save_HF0(hf0_dict, config, filename)

    model_path = os.path.join(config['repo'], config['weights_path'])
    model = load_model(model_path)
    pitch_estimates = get_prediction(np.array(hf0_dict['HF0']), model)
    
    output_folder = '{0}/{1}'.format(
        config['output_folder'], 
        config['melody_folder']
    )
    output_file = filename.split('.wav')[0] + '.csv'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    save_output(pitch_estimates, os.path.join(output_folder, output_file))
    return pitch_estimates


def run_baseline(config):
    for filename in os.listdir(config['input_data']):
        run_baseline_single_call(config, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logging-level", type=str, default="WARNING")
    args = parser.parse_args()

    config = load_config(args.config)
    run_baseline(config['baseline'])