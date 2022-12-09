import os
import argparse

import pandas as pd
import numpy as np
import mir_eval

from utils import load_config, hertz2weimar, comparative_pianoroll, weimar2hertz
from dataset import WeimarDB
from tqdm import tqdm

### delete
def get_predictions_deprecated(config, sample, scale = 'weimar', filter = True):
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

def get_baseline_evaluation_result(config, baseline = 'melodia'):
    if baseline == 'melodia':
        path = os.path.join(
            config['shared']['exp_folder'],
            config['baseline_evaluation']['output_folder'],
            config['baseline_evaluation']['b2_output_file']
        )
    else:
        path = os.path.join(
            config['shared']['exp_folder'],
            config['baseline_evaluation']['output_folder'],
            config['baseline_evaluation']['b1_output_file']
        )
    return pd.read_csv(path)

def get_output_filename_for_pianoroll(config, sample):
    filename = os.path.join(
        config['shared']['exp_folder'],
        config['baseline_evaluation']['output_folder'],
        config['baseline_evaluation']['pianoroll_folder'],
        sample['query'] + '.png'
    )
    return filename


def make_pianoroll_plots(config):
    eval_mel = get_baseline_evaluation_result(config, 'melodia')
    eval_sfnmf = get_baseline_evaluation_result(config, 'sfnmf')
    wdb = WeimarDB(config)
    for sample in tqdm(wdb):
        sfnmf_pred = get_predictions(config, sample, baseline = 'sfnmf')
        melodia_pred = get_predictions(config, sample, baseline = 'melodia')
        mel_stats = eval_mel[eval_mel.Name == sample['query']].iloc[0].to_dict()
        sf_stats = eval_sfnmf[eval_sfnmf.Name == sample['query']].iloc[0].to_dict()
        filename = get_output_filename_for_pianoroll(config, sample)
        try:
            comparative_pianoroll(
                sample['melody'],
                sfnmf_pred,
                melodia_pred,
                output_file = filename,
                title = sample['query'],
                eval_mel = mel_stats,
                eval_sfnmf = sf_stats
            )
        except:
            print(sample['query'], ' : image is too large')


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


def evaluate_old(jazzdap_config):
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

def get_prediction_filename(config, sample, baseline = 'melodia'):
    if baseline == 'melodia':
        predictions_file = '{0}/{1}/{2}{3}'.format(
            config['shared']['exp_folder'],
            config['baseline_evaluation']['melodia_folder'],
            sample['query'].replace('#1_Solo', ''),
            config['baseline_evaluation']['melodia_suffix']
        )
    else:
        predictions_file = '{0}/{1}/{2}.csv'.format(
            config['shared']['exp_folder'],
            config['baseline_evaluation']['predictions_folder'],
            sample['query']
        )
    return predictions_file

def get_predictions(config, sample, scale = 'weimar', filter = True, baseline = 'melodia', to_int = False):
    predictions_file = get_prediction_filename(config, sample, baseline)
    predictions = pd.read_csv(predictions_file, header = None)
    predictions.columns = ['onset', 'pitch']
    if filter:
        predictions = predictions[predictions.pitch > 0]
    if scale == 'weimar':
        predictions['pitch'] = hertz2weimar(predictions['pitch'], to_int = to_int)
    return predictions


def fill_pauses(sample):
    onset = sample.melody.onset
    offset = sample.melody.onset +  sample.melody.duration
    pitch = sample.melody.pitch
    rows = []
    row = {
        'pitch': 0,
        'onset': 0,
        'voicing': 0
    }
    rows.append(row)
    for index in range(onset.shape[0]):
        row = {
            'pitch': weimar2hertz(pitch.iloc[index]),
            'onset': onset.iloc[index],
            'voicing': 1
        }
        rows.append(row)
        row = {
            'pitch': 0,
            'onset': offset.iloc[index],
            'voicing': 0
        }
        rows.append(row)
    return pd.DataFrame(rows)
    
def cut_margins(pred, ground_truth):
    pred = pred[pred.onset >= ground_truth.onset.min()]
    pred = pred[pred.onset <= ground_truth.onset.max()]
    return pred

def evaluate_single_sample(config, sample, baseline):
    predictions = get_predictions(
        config, 
        sample, 
        scale = 'hertz', 
        filter = False,
        baseline = baseline
    )
    ground_truth = fill_pauses(sample)
    predictions = cut_margins(predictions, ground_truth)
    res_gt_pitch, res_gt_times = mir_eval.melody.resample_melody_series(
        ground_truth.onset,
        ground_truth.pitch,
        ground_truth.voicing,
        predictions.onset
    )
    (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(
        np.arange(predictions.shape[0]), 
        res_gt_pitch,
        np.arange(predictions.shape[0]),
        predictions.pitch.values
    )
    vr, vfa = mir_eval.melody.voicing_measures(ref_v, est_v)
    rpa = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)
    rca = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)
    oa = mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)
    eval_result = {
        'Name': sample['query'],
        'Voicing Recall': vr,
        'Voicing False Alarm': vfa,
        'Raw Pitch Accuracy': rpa,
        'Raw Chroma Accuracy': rca,
        'Overall Accuracy': oa
    }
    return eval_result


def evaluate(config):
    wdb = WeimarDB(config)
    eval_sf_nmf_rcnn = []
    eval_melodia = []
    for sample in tqdm(wdb):
        eval_sf_nmf_rcnn.append(
            evaluate_single_sample(config, sample, baseline = 'sf_nmf_rcnn')
        )
        eval_melodia.append(
            evaluate_single_sample(config, sample, baseline = 'melodia')
        )
    output_file_sf_nmf = os.path.join(
        config['shared']['exp_folder'],
        config['baseline_evaluation']['output_folder'],
        config['baseline_evaluation']['b1_output_file']
    )
    output_file_melodia = os.path.join(
        config['shared']['exp_folder'],
        config['baseline_evaluation']['output_folder'],
        config['baseline_evaluation']['b2_output_file']
    ) 
    pd.DataFrame(eval_sf_nmf_rcnn).to_csv(output_file_sf_nmf, float_format='%.3f')
    pd.DataFrame(eval_melodia).to_csv(output_file_melodia, float_format='%.3f')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logging-level", type=str, default="WARNING")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(config)
    make_pianoroll_plots(config)