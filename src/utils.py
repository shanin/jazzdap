import os
import yaml
import numpy as np
import pandas as pd
import IPython.display as ipd
import matplotlib.pyplot as plt
import matplotlib.lines as lines


def load_config(filename):
    with open(filename) as f:
        config = yaml.safe_load(f.read())
    return config

#def pianoroll(melody):
#    plt.figure(figsize=(25,5))
#    plt.scatter(melody['onset'], melody['pitch'], marker='.')


def transcription2onehot(transcription):
    cleaned_transcription = transcription - 35
    cleaned_transcription[cleaned_transcription == -35] = 0
    mask_1 = np.tile(cleaned_transcription, (63, 1))
    mask_2 = np.tile(np.arange(63).reshape((63,1)), (1, cleaned_transcription.shape[0]))
    return (mask_1 == mask_2).astype(int)


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

def adapt_config_for_notebook(config):
    config['shared']['exp_folder'] = '../' + config['shared']['exp_folder']
    config['shared']['raw_data'] = '../' + config['shared']['raw_data']
    config['dataset']['weimardb'] = '../' + config['dataset']['weimardb']
    config['dataset']['subset'] = '../' + config['dataset']['subset']
    config['baseline']['repo'] = '../' + config['baseline']['repo']
    config['baseline_evaluation']['output_folder'] = '../' + config['baseline_evaluation']['output_folder']
    return config


def comparative_pianoroll(melody, predictions, melodia, legend = True, output_file = None, title = None, scale_factor = 0.7, height = 10, eval_mel = None, eval_sfnmf = None):
    plt.rcParams['figure.dpi'] = 200
    width = min(predictions.onset.max() * scale_factor, 400)
    fig, ax = plt.subplots(figsize=(width, height))
    
    lowest_note = int(predictions['pitch'].min())
    highest_note = int(predictions['pitch'].max())
    first_onset = predictions['onset'].min()
    last_onset = predictions['onset'].max() + 2

    for index in range(lowest_note - 1, highest_note + 1):
        if index % 12 == 11:
            color_ = 'gray'
        else:
            color_ = 'gainsboro'
        ax.plot(
            [first_onset, last_onset], 
            [index + 0.5, index + 0.5], 
            color = color_,
            linewidth = 0.4,
            label='_nolegend_'
        )

    for index in range(lowest_note, highest_note +1):
        if (index % 12) in [1, 3, 6, 8, 10]:
            ax.fill_between(
                [first_onset, last_onset], 
                [index - 0.5, index - 0.5], 
                [index + 0.5, index + 0.5], 
                color = 'whitesmoke',
                label='_nolegend_'
            )
    
    sf = ax.scatter(
        predictions['onset'], 
        predictions['pitch'], 
        marker='o', 
        color = 'lightskyblue'
    )
    gt = ax.scatter(melody['onset'], melody['pitch'], marker='.', color = 'black')
    
    for _, row in melody.iterrows():
        time = [row.onset, row.onset + row.duration]
        pitch = [row.pitch, row.pitch]
        ax.plot(time, pitch, color = 'black')

    if melodia:
        ml = ax.scatter(
            melodia['onset'], 
            melodia['pitch'], 
            marker = 'o', 
            s = 0.1, 
            color = 'orange',
            zorder=2,
            alpha = 0.6
        )
    

    if legend:
        textstr = '\n'.join([
            'MELODIA',
            'Voicing Recall: {0:.2f}'.format(eval_mel['Voicing Recall']),
            'Voicing False Alarm: {0:.2f}'.format(eval_mel['Voicing False Alarm']),
            'Raw Pitch Accuracy: {0:.2f}'.format(eval_mel['Raw Pitch Accuracy']),
            'Raw Chroma Accuracy: {0:.2f}'.format(eval_mel['Raw Chroma Accuracy']),
            'Overall Accuracy: {0:.2f}'.format(eval_mel['Overall Accuracy']),
            '',
            'SF NMF RCNN',
            'Voicing Recall: {0:.2f}'.format(eval_sfnmf['Voicing Recall']),
            'Voicing False Alarm: {0:.2f}'.format(eval_sfnmf['Voicing False Alarm']),
            'Raw Pitch Accuracy: {0:.2f}'.format(eval_sfnmf['Raw Pitch Accuracy']),
            'Raw Chroma Accuracy: {0:.2f}'.format(eval_sfnmf['Raw Chroma Accuracy']),
            'Overall Accuracy: {0:.2f}'.format(eval_sfnmf['Overall Accuracy'])
        ])
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.01, 0.3, textstr, fontsize=10, transform=ax.transAxes,
            verticalalignment='top', bbox=props)


    ax.set_yticks([24, 36, 48, 60, 72, 84])
    ax.set_yticklabels(['C2', 'C3', 'C4', 'C5', 'C6', 'C7'])

    if legend:
        plt.legend([sf, gt, ml], ['SF NMF RCNN', 'Ground Truth', 'MELODIA'], loc = 'upper left')
    plt.xlabel('seconds')
    plt.ylabel('pitch')
    plt.xlim(first_onset, last_onset)
    plt.ylim(lowest_note - 0.5, highest_note + 0.5)
    if title:
        plt.title(title, loc = 'left')
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()



def comparative_pianoroll__deprecated(melody, predictions, output_file = None, title = None, scale_factor = 0.7, height = 10):
    width = min(predictions.onset.max() * scale_factor, 400)
    fig, ax = plt.subplots(figsize=(width, height))
    
    lowest_note = int(predictions['pitch'].min())
    highest_note = int(predictions['pitch'].max())
    first_onset = predictions['onset'].min()
    last_onset = predictions['onset'].max() + 2

    for index in range(lowest_note - 1, highest_note + 1):
        if index % 12 == 11:
            color_ = 'gray'
        else:
            color_ = 'gainsboro'
        ax.plot(
            [first_onset, last_onset], 
            [index + 0.5, index + 0.5], 
            color = color_,
            linewidth = 0.4,
            label='_nolegend_'
        )

    for index in range(lowest_note, highest_note +1):
        if (index % 12) in [1, 3, 6, 8, 10]:
            ax.fill_between(
                [first_onset, last_onset], 
                [index - 0.5, index - 0.5], 
                [index + 0.5, index + 0.5], 
                color = 'whitesmoke',
                label='_nolegend_'
            )
    
    ax.scatter(predictions['onset'], predictions['pitch'], marker='o', color = 'lightskyblue')
    ax.scatter(melody['onset'], melody['pitch'], marker='.', color = 'black')
    
    for _, row in melody.iterrows():
        time = [row.onset, row.onset + row.duration]
        pitch = [row.pitch, row.pitch]
        ax.plot(time, pitch, color = 'black')

    ax.set_yticks([24, 36, 48, 60, 72, 84])
    ax.set_yticklabels(['C2', 'C3', 'C4', 'C5', 'C6', 'C7'])

    plt.legend(['SF NMF RCNN', 'ground truth'])
    plt.xlabel('seconds')
    plt.ylabel('pitch')
    plt.xlim(first_onset, last_onset)
    plt.ylim(lowest_note - 0.5, highest_note + 0.5)
    if title:
        plt.title(title)
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()



def pianoroll(melody, predictions = None, beats = None, output_file = None, title = None, scale_factor = 0.7, height = 10):
    width = min(melody.onset.max() * scale_factor, 400)
    fig, ax = plt.subplots(figsize=(width, height))
    
    lowest_note = int(melody['pitch'].min())
    highest_note = int(melody['pitch'].max())
    first_onset = melody['onset'].min() - 2
    last_onset = melody['onset'].max() + 2

    for index in range(lowest_note - 1, highest_note + 1):
        if index % 12 == 11:
            color_ = 'gray'
        else:
            color_ = 'gainsboro'
        ax.plot(
            [first_onset, last_onset], 
            [index + 0.5, index + 0.5], 
            color = color_,
            linewidth = 0.4,
            label='_nolegend_'
        )

    for index in range(lowest_note, highest_note +1):
        if (index % 12) in [1, 3, 6, 8, 10]:
            ax.fill_between(
                [first_onset, last_onset], 
                [index - 0.5, index - 0.5], 
                [index + 0.5, index + 0.5], 
                color = 'whitesmoke',
                label='_nolegend_'
            )

    ax.scatter(melody['onset'], melody['pitch'], marker='.', color = 'black')
    
    for _, row in melody.iterrows():
        time = [row.onset, row.onset + row.duration]
        pitch = [row.pitch, row.pitch]
        ax.plot(time, pitch, color = 'black')

    ax.set_yticks([24, 36, 48, 60, 72, 84])
    ax.set_yticklabels(['C2', 'C3', 'C4', 'C5', 'C6', 'C7'])

    #plt.legend(['SF NMF RCNN', 'ground truth'])
    plt.xlabel('seconds')
    plt.ylabel('pitch')
    plt.xlim(first_onset, last_onset)
    plt.ylim(lowest_note - 0.5, highest_note + 0.5)
    if title:
        plt.title(title)
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

def play_audio(sample):
    print(sample['query'])
    return ipd.Audio(rate = sample['sample_rate'], data = sample['audio'])

def weimar2hertz(n):
    return 440 * (2 ** ((n - 69) / 12))

def hertz2weimar(f, to_int = True):
    value = 12 * np.log2(f/440) + 69
    if to_int:
        return round(value)
    else:
        return value

def safe_mkdir(path):
    current_path = os.getcwd()
    for elem in path.split('/'):
        current_path = os.path.join(current_path, elem)
        if not os.path.exists(current_path):
            os.mkdir(current_path)