import os
import yaml
import numpy as np
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

def comparative_pianoroll(melody, predictions, output_file = None, title = None, scale_factor = 0.7, height = 10):
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



def pianoroll(melody, output_file = None, title = None, scale_factor = 0.7, height = 10):
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

def hertz2weimar(f):
    return np.round(12 * np.log2(f/440)) + 69

def safe_mkdir(path):
    current_path = os.getcwd()
    for elem in path.split('/'):
        current_path = os.path.join(current_path, elem)
        if not os.path.exists(current_path):
            os.mkdir(current_path)