import os
import argparse

import pandas as pd
import numpy as np
import mir_eval

from utils import load_config, hertz2weimar, comparative_pianoroll, weimar2hertz
from dataset import WeimarDB


def get_predictions(config, sample, scale = 'weimar', filter = True):
    predictions_file = '{0}/{1}/{2}.csv'.format(
        config['shared']['exp_folder'],
        config['baseline_evaluation']['predictions_folder'],
        sample['query']
    )
    predictions = pd.read_csv(predictions_file, header = None)
    predictions.columns = ['onset', 'pitch']
    if filter:
        predictions = predictions[predictions.pitch > 0]
    if scale == 'weimar':
        predictions['pitch'] = hertz2weimar(predictions['pitch'])
    return predictions
    

def make_pianoroll_comparison(config, sample):
    config_local = config['baseline_evaluation']
    config_shared = config['shared']
    predictions = get_predictions(config, sample)
    output_folder = os.path.join(
        config_shared['exp_folder'],
        config_local['output_folder']
    )
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    pianoroll_folder = os.path.join(
        output_folder, 
        config_local['pianoroll_folder']
    )
    if not os.path.exists(pianoroll_folder):
        os.mkdir(pianoroll_folder)
    output_file = '{0}/{1}.png'.format(
        pianoroll_folder,
        sample['query']
    )
    comparative_pianoroll(
        sample['melody'], 
        predictions, 
        output_file = output_file,
        title = sample['query']
    )


def make_pianoroll_plots(config):
    wdb = WeimarDB(config)
    for idx in range(len(wdb)):
        sample = wdb[idx]
        make_pianoroll_comparison(config, sample)

def get_quantized_labels(sample, predictions):
    time_steps = predictions.onset
    melody = sample['melody']
    rows = []
    for step in time_steps:
        if melody[melody.onset < step].shape[0] > 0:
            last_onset = melody[melody.onset < step].iloc[-1,:]
            if last_onset.onset + last_onset.duration > step:
                row = {
                    'onset': step,
                    'pitch': weimar2hertz(last_onset.pitch)
                }
            else:
                row = {
                    'onset': step,
                    'pitch': -1
                }
        else:
            row = {
                'onset': step,
                'pitch': -1
            }
        rows.append(row)
    labels = pd.DataFrame(rows)
    return labels

def cut_unvoiced_frames(labels, pitch_estimates):
    start = np.argwhere(labels != -1).flatten().min()
    stop = np.argwhere(labels != -1).flatten().max()
    return labels[start:stop], pitch_estimates[start:stop]

def melody_evaluation(config, sample, cut = True):

    pitch_estimates = get_predictions(config, sample, scale = 'hertz', filter = False)
    labels = get_quantized_labels(sample, pitch_estimates)

    pitch_estimates = np.array(pitch_estimates.iloc[:,1])
    labels = np.array(labels.iloc[:,1])

    if cut:
        labels, pitch_estimates = cut_unvoiced_frames(labels, pitch_estimates)

    evaluation_results = {}

    (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(np.arange(np.size(labels)),
                                                                   labels,
                                                                   np.arange(np.size(labels)),
                                                                   pitch_estimates)

    vr, vfa = mir_eval.melody.voicing_measures(ref_v, est_v)
    rpa = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)
    rca = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)
    oa = mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)

    evaluation_results['query'] = sample['query']
    evaluation_results['Voicing Recall'] = vr
    evaluation_results['Voicing False Alarm'] = vfa
    evaluation_results['Raw Pitch Accuracy'] = rpa
    evaluation_results['Raw Chroma Accuracy'] = rca
    evaluation_results['Overall Accuracy'] = oa
    return evaluation_results


def evaluate(jazzdap_config):
    dataset_config = jazzdap_config['dataset']
    eval_config = jazzdap_config['baseline_evaluation']
    wdb = WeimarDB(dataset_config)
    rows = []
    for idx in range(len(wdb)):
        sample = wdb[idx]
        eval_results = melody_evaluation(eval_config, sample)
        rows.append(eval_results)
        print(eval_results)
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(eval_config['output_folder'], eval_config['output_file']))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logging-level", type=str, default="WARNING")
    args = parser.parse_args()

    config = load_config(args.config)
    make_pianoroll_plots(config)
    evaluate(config)