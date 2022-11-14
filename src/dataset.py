import os
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample
from math import floor
import sqlite3
import IPython.display as ipd
import matplotlib.pyplot as plt
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor

WDB_SIZE = 456

MISTAKES = {
    'DickieWells_Jo-Jo_Orig': 'DickieWells_Jo=Jo_Orig',
    "BobBerg_ IDidn'tKnowWhatTimeItWas_Orig": "BobBerg_IDidn'tKnowWhatTimeItWas_Orig",
    "Bob Berg_SecondSightEnterTheSpirit_Orig": "BobBerg_SecondSightEnterTheSpirit_Orig",
    "DexterGordon_Society Red_Orig": "DexterGordon_SocietyRed_Orig",
    "DizzyGillespie_Be-Bop_Orig": "DizzyGillespie_Be=Bop_Orig",
    "Don Byas_OutOfNowhere_Orig": "DonByas_OutOfNowhere_Orig",
    "JohnColtrane_26-2_Orig": "JohnColtrane_26=2_Orig",
    "SonnyStitt_BluesInBe-Bop_Orig": "SonnyStitt_BluesInBe=Bop_Orig",
    "LesterYoung_D.B. Blues_Orig": "LesterYoung_D.B.Blues_Orig",
    "MilesDavis_Eighty-One_Orig": "MilesDavis_Eighty=One_Orig",
    "Dave Holland_TakeTheColtrane_Orig": "DaveHolland_TakeTheColtrane_Orig"
}

SOLOSTART_CORRECTIONS = {
    43: 29.9,
    64: -15.345,
    79: 93.1722,
    82: -93.1722,
    171: -64.85,
    309: -212.5,
    382: 205.9
}

class WeimarSolo(object):
    def __init__(self):
        self.melid = None
        self.trackid = None
        self.filename = None
        self.audio = None
        self.solostart = None
        self.solostop = None
        self.melody = None
        self.beats = None
        self.sample_rate = None
        self.resample_rate = None
        self.predicted_beats = None

    def __str__(self):
        return(self.filename)
    
    def __repr__(self):
        return(self.filename)

    def resampled_audio(self):   
        transform = Resample(
            orig_freq = self.sample_rate, 
            new_freq = self.resample_rate
        )
        return transform(self.audio)

    def mono_audio(self):
        return self.audio.mean(axis = 0)

    def predict_beats(self):
        activations = RNNBeatProcessor()(self.mono_audio().numpy())
        self.predicted_beats = BeatTrackingProcessor(fps=100)(activations)
        return self.predicted_beats

    def export_to_sv(self):
        pd.DataFrame(
            {
                'onset': self.melody.onset + self.solostart,
                'duration': self.melody.duration,
                'pitch': self.melody.pitch
            }
        ).to_csv(self.filename + '.melody.csv', index = False)

    def play(self):
        print(self.filename)
        return ipd.Audio(rate = self.sample_rate, data = self.audio)

    def pianoroll(self, predictions = None, beats = None, 
            output_file = None, title = None, scale_factor = 0.7, height = 10):

        width = min(self.melody.onset.max() * scale_factor, 400)
        _, ax = plt.subplots(figsize=(width, height))
        
        lowest_note = int(self.melody.pitch.min())
        highest_note = int(self.melody.pitch.max())
        first_onset = self.melody.onset.min() - 2
        last_onset = self.melody.onset.max() + 2

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
                    label='_nolegend_'
                )

        #ground truth onset plot
        ax.scatter(self.melody.onset, self.melody.pitch, marker='.', color = 'black')
        
        #plot note duration
        for _, row in self.melody.iterrows():
            time = [row.onset, row.onset + row.duration]
            pitch = [row.pitch, row.pitch]
            ax.plot(time, pitch, color = 'black')

        ax.set_yticks([24, 36, 48, 60, 72, 84])
        ax.set_yticklabels(['C2', 'C3', 'C4', 'C5', 'C6', 'C7'])

        plt.xlabel('seconds')
        plt.ylabel('pitch')
        plt.xlim(first_onset, last_onset)
        plt.ylim(lowest_note - 0.5, highest_note + 0.5)
        plt.title(self.__repr__())
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            plt.close()

            

class WeimarDB(Dataset):
    def __init__(self, config):
        config_shared = config['shared']
        config_local = config['dataset']

        self._audio_dir = os.path.join(
            config_shared['raw_data'], 
            config_local['audio_dir']
        )

        self._resample_rate = config_local.get('resample_rate', None)
        self.load_beats = config_local.get('load_beats', False)
        self._init_column_names()
        self._init_database_cursor(config_local)
        self._init_idx_dict(config_local)


    def _init_idx_dict(self, config):
        if config['full']:
            self._idx_dict = [elem for elem in range(1, WDB_SIZE + 1)]
        else:
            with open(config['subset'], 'r') as file:
                line = file.readline()
            self._idx_dict = [int(elem) for elem in line.split(',')]


    def _init_column_names(self):
        self._melody_columns = [
            'eventid', 'melid', 'onset', 'pitch', 'duration', 'period', 
            'division', 'bar', 'beat', 'tatum', 'subtatum', 'num', 'denom', 
            'beatprops', 'beatdur', 'tatumprops', 'loud_max', 'loud_med', 
            'loud_sd', 'loud_relpos', 'loud_cent', 'loud_s2b', 'f0_mod', 
            'f0_range', 'f0_freq_hz', 'f0_med_dev'
        ]
        self._solo_info_columns = [
            'melid', 'trackid', 'compid', 'recordid', 'performer', 'title', 
            'titleaddon', 'solopart', 'instrument', 'style', 'avgtempo', 
            'tempoclass', 'rhythmfeel', 'key', 'signature', 'chord_changes', 
            'chorus_count'
        ]
        self._transcription_info_columns = [
            'melid', 'trackid', 'filename_sv', 'filename_solo', 
            'solotime', 'solostart_sec', 'status'
        ]
        self._track_info_columns = [
            'trackid', 'compid', 'recordid', 'filename_track', 'lineup',
            'mbzid', 'trackno', 'recordingdate'
        ]
        self._beats_columns = [
            'beatid', 'melid', 'onset', 'bar', 'beat', 'signature', 'chord',
            'form', 'bass_pitch', 'chorus_id'
        ]


    def _init_database_cursor(self, config):
        self._connect = sqlite3.connect(config['weimardb'])
        self._cursor = self._connect.cursor()


    def _parse_transcription_info(self, solo):
        colnames_str = ', '.join(self._transcription_info_columns)
        query = f'SELECT {colnames_str} FROM transcription_info WHERE melid = {solo.melid}'
        transcription_info = pd.DataFrame(
            self._cursor.execute(query).fetchall(), 
            columns = self._transcription_info_columns
        )
        solo.trackid = transcription_info.trackid.values[0]
        solo.solostart = max(transcription_info.solostart_sec.values[0], 0)
        solo.solostart += SOLOSTART_CORRECTIONS.get(solo.melid, 0)
        solo.solostop = solo.solostart + solo.melody.onset.max() + 2


    def _parse_track_info(self, solo):
        colnames_str = ', '.join(self._track_info_columns)
        query = f'SELECT {colnames_str} FROM track_info WHERE trackid = {solo.trackid}'
        track_info = pd.DataFrame(
            self._cursor.execute(query).fetchall(), 
            columns = self._track_info_columns
        )
        filename = track_info.filename_track.values[0]
        for key in MISTAKES:
            filename = filename.replace(key, MISTAKES[key])
        solo.filename = filename
    

    def _parse_melody(self, solo):
        colnames_str = ', '.join(self._melody_columns)
        query = f'SELECT {colnames_str} FROM melody WHERE melid = {solo.melid}'
        solo.melody = pd.DataFrame(
            self._cursor.execute(query).fetchall(), 
            columns = self._melody_columns
        )


    def _parse_beats(self, solo):
        colnames_str = ', '.join(self._beats_columns)
        query = f'SELECT {colnames_str} FROM beats WHERE melid = {solo.melid}'
        solo.beats = pd.DataFrame(
            self._cursor.execute(query).fetchall(), 
            columns = self._beats_columns
        )


    def _load_audio(self, solo):
        audio, solo.sample_rate = torchaudio.load(
            os.path.join(
                self._audio_dir, 
                f'{solo.filename}.wav'
            )
        )
        sample_start = floor(solo.solostart * solo.sample_rate)
        if solo.solostop:
            sample_stop = floor(solo.solostop * solo.sample_rate)
        else:
            sample_stop = -1
        solo.audio = audio[:, sample_start : sample_stop]


    def __getitem__(self, idx):
        
        solo = WeimarSolo()
        solo.melid = self._idx_dict[idx]
        self._parse_melody(solo)
        self._parse_beats(solo)
        self._parse_transcription_info(solo)
        self._parse_track_info(solo)
        solo.resample_rate = self._resample_rate
        self._load_audio(solo)
        return solo
        

    def __len__(self):
        return WDB_SIZE