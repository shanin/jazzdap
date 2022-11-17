import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample
from math import floor
import sqlite3
import IPython.display as ipd
import matplotlib.pyplot as plt
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
from mir_eval.melody import resample_melody_series
import warnings
from sklearn.preprocessing import LabelBinarizer
from utils import transcription2onehot

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
    "Dave Holland_TakeTheColtrane_Orig": "DaveHolland_TakeTheColtrane_Orig",
    "ChrisPotter_InASentimentalMood_Orig": "ChrisPotter_InaSentimentalMood_Orig",
    "MilesDavis_Gingerbreadboy_Orig": "MilesDavis_GingerbreadBoy_Orig" 
}

SOLO_MISTAKES = {
    "FatsNavarro_GoodBait_No1_Solo": "FatsNavarro_GoodBait_Solo",
    "FatsNavarro_GoodBait_No2_Solo": "FatsNavarro_GoodBait_AlternateTake_Solo",
    "BranfordMarsalis_Ummg_Solo": "BranfordMarsalis_U.M.M.G._Solo",
    "SonnyRollins_I'llRememberApril-AlternateTake2_Solo": "SonnyRollins_I'llRememberApril_AlternateTake2_Solo",
    "PaulDesmond_BlueRondoAlaTurk_Solo": "PaulDesmond_BlueRondoALaTurk_Solo"
}

SOLO_PATCH_FILES = ['LouisArmstrong_CornetChopSuey_Solo']

SOLOSTART_CORRECTIONS = {
    43: 29.9,
    64: -15.345,
    79: 93.1722,
    82: -93.1722,
    171: -64.85,
    309: -212.5,
    382: 205.9
}

TRAIN_ARTISTS = [
    'Warne Marsh', 'Benny Carter', 'Joe Lovano', 'Bix Beiderbecke',
    'Von Freeman', 'Don Ellis', 'John Coltrane', 'Wynton Marsalis',
    'Chet Baker', 'Clifford Brown', 'Lee Konitz', 'Charlie Shavers',
    'Pat Martino', 'Harry Edison', 'Cannonball Adderley',
    'Coleman Hawkins', 'George Coleman', 'Curtis Fuller',
    'Charlie Parker', 'Lee Morgan', 'Roy Eldridge', 'Sonny Stitt',
    'Chu Berry', 'Louis Armstrong', 'Art Pepper', 'Branford Marsalis',
    'Pat Metheny', 'Steve Lacy', 'Nat Adderley', 'Buck Clayton',
    'Milt Jackson', 'Red Garland', 'Zoot Sims', 'Lester Young',
    'Dizzy Gillespie', 'John Abercrombie', 'Johnny Dodds',
    'Ornette Coleman', 'J.J. Johnson', 'Sidney Bechet',
    'Kenny Wheeler', 'Johnny Hodges', 'Rex Stewart',
    'J.C. Higginbotham', 'Miles Davis', 'Chris Potter', 'Pepper Adams',
    'Fats Navarro', 'Steve Turre', 'Gerry Mulligan', 'Michael Brecker',
    'Herbie Hancock', 'Dickie Wells', 'Joe Henderson', 'Sonny Rollins',
    'Woody Shaw', 'Eric Dolphy', 'Kai Winding', 'Phil Woods',
    'Wayne Shorter', 'Kenny Garrett', 'David Murray', 'Lionel Hampton',
    'Bob Berg'
]

TEST_OLD_ARTISTS = [
    'Henry Allen', 'Dexter Gordon', 'Kenny Dorham', 'Benny Goodman',
    'Paul Desmond', 'Don Byas', 'Ben Webster', 'Hank Mobley',
    'Stan Getz', 'David Liebman', 'Steve Coleman', 'Kid Ory',
    'Joshua Redman', 'Freddie Hubbard'
]

TEST_ARTISTS = [
    'Ben Webster', 'Cannonball Adderley', 'Charlie Shavers', 'Chu Berry', 
    'Chris Potter', 'Dexter Gordon', 'Fats Navarro', 'Joe Henderson', 
    'John Coltrane', 'Kenny Wheeler', 'Paul Desmond', 'Sidney Bechet', 
    'Sonny Rollins', 'Wynton Marsalis'
]


INSTRUMENTS = [
    'cl', 'as', 'ts', 'cor', 'tp', 'tb', 'ss', 'bcl', 'bs', 'p',
    'ts-c', 'g', 'vib'
]

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
        self.performer = None
        self.title = None
        self.instrument = None
        self.sfnmf = None

    def __str__(self):
        return(f'{self.performer} ({self.instrument}) - {self.title} (from {self.filename})')
    
    def __repr__(self):
        return(f'{self.performer} ({self.instrument}) - {self.title} (from {self.filename})')

    def fill_pauses(self):
        onset = self.melody.onset
        offset = self.melody.onset +  self.melody.duration
        pitch = self.melody.pitch
        rows = []
        row = {
            'pitch': 0,
            'onset': 0,
            'voicing': 0
        }
        rows.append(row)
        for index in range(onset.shape[0]):
            row = {
                'pitch': pitch.iloc[index],
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

    def resampled_audio(self):   
        transform = Resample(
            orig_freq = self.sample_rate, 
            new_freq = self.resample_rate
        )
        return transform(self.audio)

    def resampled_transcription(self, onehot = True):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ground_truth = self.fill_pauses()
            times_new = np.arange(self.sfnmf.shape[1]) * 0.0116 
            res_gt_pitch, res_gt_times = resample_melody_series(
                ground_truth.onset,
                ground_truth.pitch,
                ground_truth.voicing,
                times_new
            )
            if onehot:
                return transcription2onehot(res_gt_pitch)
            else:
                return res_gt_pitch
        

    def mono_audio(self):
        return self.audio.mean(axis = 0)

    def predict_beats(self):
        activations = RNNBeatProcessor()(self.mono_audio().numpy())
        self.predicted_beats = BeatTrackingProcessor(fps=100)(activations)
        return self.predicted_beats

    def export_to_sv(self):
        pd.DataFrame(
            {
                'onset': self.melody.onset,
                'duration': self.melody.duration,
                'pitch': self.melody.pitch
            }
        ).to_csv(self.filename_solo + '.melody.csv', index = False)

    def play(self):
        print(self.__repr__())
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
    def __init__(self, config, partition = 'full', instrument = 'any', autoload_audio = True, autoload_sfnmf = False):
        config_shared = config['shared']
        config_local = config['dataset']

        self._patch_dir = os.path.join(
            config_shared['raw_data'], 
            config_local['patch_dir']
        )

        self._audio_dir = os.path.join(
            config_shared['raw_data'], 
            config_local['audio_dir']
        )

        self._sfnmf_dir = os.path.join(
            config_shared['exp_folder'],
            config_local['sfnmf_dir']
        )

        self.autoload_audio = autoload_audio
        self.autoload_sfnmf = autoload_sfnmf
        self._resample_rate = config_local.get('resample_rate', None)
        self.load_beats = config_local.get('load_beats', False)
        self._init_column_names()
        self._init_database_cursor(config_local)
        self._init_melid_list(partition, instrument)

    def _init_melid_list(self, partition, instrument):
        solo_info = self.get_solo_info()
        if partition == 'train':
            solo_info = solo_info[solo_info.performer.isin(TRAIN_ARTISTS)]
        elif partition == 'test':
            solo_info = solo_info[solo_info.performer.isin(TEST_ARTISTS)]
        if instrument in INSTRUMENTS:
            solo_info = solo_info[solo_info.instrument == instrument]
        self._melid_list = solo_info.melid.values


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
        solo.filename_solo = transcription_info.filename_solo.values[0]
        for key in SOLO_MISTAKES:
            solo.filename_solo = solo.filename_solo.replace(key, SOLO_MISTAKES[key])

    def _parse_solo_info(self, solo):
        colnames_str = ', '.join(self._solo_info_columns)
        query = f'SELECT {colnames_str} FROM solo_info WHERE melid = {solo.melid}'
        solo_info = pd.DataFrame(
            self._cursor.execute(query).fetchall(), 
            columns = self._solo_info_columns
        )
        solo.instrument = solo_info.instrument.values[0]
        solo.performer = solo_info.performer.values[0]
        solo.title = solo_info.title.values[0]


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
        if solo.filename_solo in SOLO_PATCH_FILES:
            audio_dir = self._patch_dir
        else:
            audio_dir = self._audio_dir
        solo.audio, solo.sample_rate = torchaudio.load(
            os.path.join(
                audio_dir, 
                f'{solo.filename_solo}.wav'
            )
        )
        #sample_start = floor(solo.solostart * solo.sample_rate)
        #if solo.solostop:
        #    sample_stop = floor(solo.solostop * solo.sample_rate)
        #else:
        #    sample_stop = -1
        #solo.audio = audio

    def _load_sfnmf(self, solo):
        path = os.path.join(
            self._sfnmf_dir,
            f'melid_{str(solo.melid).zfill(3)}.npy'
        )
        solo.sfnmf = np.load(path)

    def get_solo_info(self):
        colnames_str = ', '.join(self._solo_info_columns)
        query = f'SELECT {colnames_str} FROM solo_info'
        return pd.DataFrame(
            self._cursor.execute(query).fetchall(), 
            columns = self._solo_info_columns
        )

    def get_transcription_info(self):
        colnames_str = ', '.join(self._transcription_info_columns)
        query = f'SELECT {colnames_str} FROM transcription_info'
        return pd.DataFrame(
            self._cursor.execute(query).fetchall(), 
            columns = self._transcription_info_columns
        )

    def __getitem__(self, idx):
        
        solo = WeimarSolo()
        solo.melid = self._melid_list[idx]
        self._parse_melody(solo)
        self._parse_beats(solo)
        self._parse_transcription_info(solo)
        self._parse_track_info(solo)
        self._parse_solo_info(solo)
        solo.resample_rate = self._resample_rate
        if self.autoload_audio:
            self._load_audio(solo)
        if self.autoload_sfnmf:
            self._load_sfnmf(solo)
        return solo
        
    def __len__(self):
        return len(self._melid_list)