import matplotlib.pyplot as plt

def boxplot():
    raise NotImplementedError



def pianoroll(sample, predictions = None, beats = None, 
        output_file = None, title = None, scale_factor = 0.7, height = 10):

    width = min(sample.melody.onset.max() * scale_factor, 400)
    _, ax = plt.subplots(figsize=(width, height))
        
    lowest_note = int(sample.melody.pitch.min())
    highest_note = int(sample.melody.pitch.max())
    first_onset = sample.melody.onset.min() - 2
    last_onset = sample.melody.onset.max() + 2

    #octave boundaries
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

    #pianoroll background
    for index in range(lowest_note, highest_note +1):
        if (index % 12) in [1, 3, 6, 8, 10]:
            ax.fill_between(
                [first_onset, last_onset], 
                [index - 0.5, index - 0.5], 
                [index + 0.5, index + 0.5], 
                color = 'whitesmoke',
                label = '_nolegend_'
            )

    #ground truth onset plot
    ax.scatter(sample.melody.onset, sample.melody.pitch, marker='.', color = 'black')
        
    #plot note duration
    for _, row in sample.melody.iterrows():
        time = [row.onset, row.onset + row.duration]
        pitch = [row.pitch, row.pitch]
        ax.plot(time, pitch, color = 'black')

    ax.set_yticks([24, 36, 48, 60, 72, 84])
    ax.set_yticklabels(['C2', 'C3', 'C4', 'C5', 'C6', 'C7'])

    plt.xlabel('seconds')
    plt.ylabel('pitch')
    plt.xlim(first_onset, last_onset)
    plt.ylim(lowest_note - 0.5, highest_note + 0.5)
    plt.title(sample.__repr__())
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

