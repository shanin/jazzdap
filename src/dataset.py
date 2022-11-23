import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import torchaudio
from torchaudio.transforms import Resample
from math import floor
import sqlite3
import IPython.display as ipd
import matplotlib.pyplot as plt
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
from mir_eval.melody import resample_melody_series
import warnings
from utils import transcription2onehot
from sklearn.preprocessing import normalize
from tqdm import tqdm

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
    "PaulDesmond_BlueRondoAlaTurk_Solo": "PaulDesmond_BlueRondoALaTurk_Solo",
    "BranfordMarsalis_GutbucketSteepy_Solo": "BranfordMarsalis_GutBucketSteepy_Solo",
    "DizzyGillespie_Blue'NBoogie_Solo": "DizzyGillespie_Blue'nBoogie_Solo",
    "EricDolphy_Aisha_solo": "EricDolphy_Aisha_Solo",
    "KidOry_Who'sit_Solo": "KidOry_Who'sIt_Solo",
    "WayneShorter_JuJu_Solo": "WayneShorter_Juju_Solo"
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
    'Art Pepper', 'Benny Carter', 'Benny Goodman', 'Bix Beiderbecke', 
    'Bob Berg', 'Branford Marsalis', 'Buck Clayton', 'Charlie Parker', 
    'Chet Baker', 'Clifford Brown', 'Coleman Hawkins', 'Curtis Fuller', 
    'David Liebman', 'David Murray', 'Dickie Wells', 'Dizzy Gillespie', 
    'Don Byas', 'Don Ellis', 'Eric Dolphy', 'Freddie Hubbard', 
    'Gerry Mulligan', 'Hank Mobley', 'Harry Edison', 
    'Henry Allen', 'Herbie Hancock', 'J.C. Higginbotham', 'J.J. Johnson', 
    'Joe Lovano', 'John Abercrombie', 'Johnny Dodds', 'Johnny Hodges', 
    'Joshua Redman', 'Kai Winding', 'Kenny Dorham', 'Kenny Garrett', 
    'Kid Ory', 'Lee Konitz', 'Lee Morgan', 'Lester Young', 'Lionel Hampton', 
    'Louis Armstrong', 'Michael Brecker', 'Miles Davis', 'Milt Jackson', 
    'Ornette Coleman', 'Pat Martino', 'Pat Metheny', 
    'Pepper Adams', 'Red Garland', 'Rex Stewart', 
    'Roy Eldridge', 'Sonny Stitt', 'Stan Getz', 'Steve Coleman', 
    'Steve Lacy', 'Steve Turre', 'Von Freeman', 'Warne Marsh', 
    'Wayne Shorter', 'Woody Shaw', 'Zoot Sims'
]

VAL_ARTISTS = [
    'Nat Adderley', 'George Coleman', 'Phil Woods'
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
        eps = 0.001
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
        prev_offset = -1
        for index in range(onset.shape[0]):
            if np.abs(prev_offset - onset.iloc[index]) < eps:
                rows.pop()
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
            prev_offset = offset.iloc[index]
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
                    label = '_nolegend_'
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
    def __init__(
            self, 
            config, 
            partition = 'full', 
            instrument = 'any', 
            performer = None,
            autoload_audio = True, 
            autoload_sfnmf = False
        ):
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
        self._init_melid_list(partition, instrument, performer)

    def _init_melid_list(self, partition, instrument, performer):
        solo_info = self.get_solo_info()
        if partition == 'train':
            solo_info = solo_info[solo_info.performer.isin(TRAIN_ARTISTS)]
        elif partition == 'test':
            solo_info = solo_info[solo_info.performer.isin(TEST_ARTISTS)]
        elif partition == 'val':
            solo_info = solo_info[solo_info.performer.isin(VAL_ARTISTS)]
        if instrument in INSTRUMENTS:
            solo_info = solo_info[solo_info.instrument == instrument]
        if performer:
            solo_info = solo_info[solo_info.performer == performer]
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



class WeimarSlicer(Dataset):

    def _parse_config(self, config):
        self.segment_length = config['crnn']['segment_length']
        self.number_of_patches = config['crnn']['number_of_patches']
        self.patch_size = config['crnn']['patch_size']
        self.feature_size = config['crnn']['feature_size']
        self.number_of_classes = config['crnn']['number_of_classes']
        self.sampling_rate = config['sfnmf']['Fs']
        self.hop = config['sfnmf']['hop']


    def _cut(self, sample, sfnmf, labels):
        first_onset = sample.melody.onset.iloc[0]
        last_offset = sample.melody.onset.iloc[-1] + sample.melody.duration.iloc[-1]
        start = floor(first_onset * (self.sampling_rate / self.hop))
        stop = floor(last_offset * (self.sampling_rate / self.hop)) + 1
        return sfnmf[:, start:stop], labels[:, start:stop]
    

    def _prepare_HF0_tensor(self, sfnmf):
        length_of_sequence = sfnmf.shape[1]
        number_of_segments = int(floor(length_of_sequence/self.segment_length))

        HF0 = np.append(
            sfnmf[:, :number_of_segments * self.segment_length],
            sfnmf[:, -self.segment_length:], 
                axis=1
        )
        HF0 = normalize(HF0, norm='l1', axis=0)
        HF0 = HF0.T

        number_of_samples = int(HF0.shape[0] / (self.number_of_patches * self.patch_size))
        HF0 = np.reshape(
            HF0, 
            (number_of_samples, 1, self.number_of_patches, self.patch_size, self.feature_size)
        )
        return torch.tensor(HF0, dtype=torch.float), length_of_sequence, number_of_samples


    def _prepare_labels_tensor(self, labels):
        length_of_sequence = labels.shape[1]
        number_of_segments = int(floor(length_of_sequence/self.segment_length))

        y = np.append(
            labels[:, :(number_of_segments * self.segment_length)],
            labels[:, -self.segment_length: ], 
            axis=1
        )
        y = y.T

        number_of_samples = int(y.shape[0] / (self.number_of_patches * self.patch_size))
        y = np.reshape(
            y,
            (number_of_samples, self.number_of_patches, self.patch_size, self.number_of_classes)
        )
        return torch.tensor(y, dtype=torch.float), length_of_sequence, number_of_samples

    def _assemble_tensors(self):
        
        X_list_of_tensors = []
        y_list_of_tensors = []
        track_lengths = []
        number_of_samples = []

        for sample in tqdm(self.dataset):
            sfnmf = sample.sfnmf
            labels = sample.resampled_transcription(onehot=True)
            sfnmf, labels = self._cut(sample, sfnmf, labels)
            
            HF0_tensor, hf0_len, samples = self._prepare_HF0_tensor(sfnmf)
            labels_tensor, _, _ = self._prepare_labels_tensor(labels)

            X_list_of_tensors.append(HF0_tensor)
            y_list_of_tensors.append(labels_tensor)
            track_lengths.append(hf0_len)
            number_of_samples.append(samples)

        self.X = torch.cat(X_list_of_tensors, dim = 0)
        self.y = torch.cat(y_list_of_tensors, dim = 0)
        self.track_lengths = track_lengths
        self.number_of_samples = number_of_samples


    def __init__(self, dataset, config, tag = None):

        self._parse_config(config)
        self.tag = tag
        self.dataset = dataset

        if self.tag is None:
            self._assemble_tensors()
        else:
            #use cached version (or save generated version to cache)
            self.cache_folder = os.path.join(
                config['shared']['exp_folder'],
                config['dataset']['cache_folder'],
                self.tag
            )
            if os.path.exists(self.cache_folder):
                self.X = torch.load(os.path.join(self.cache_folder, 'X.pt'))
                self.y = torch.load(os.path.join(self.cache_folder, 'y.pt'))
                self.track_lengths = \
                    torch.load(os.path.join(self.cache_folder, 'track_lengths.pt'))
                self.number_of_samples = \
                    torch.load(os.path.join(self.cache_folder, 'number_of_samples.pt'))
            else:
                self._assemble_tensors()
                os.makedirs(self.cache_folder, exist_ok = True)
                torch.save(self.X, os.path.join(self.cache_folder, 'X.pt'))
                torch.save(self.y, os.path.join(self.cache_folder, 'y.pt'))
                torch.save(self.track_lengths, os.path.join(self.cache_folder, 'track_lengths.pt'))
                torch.save(self.number_of_samples, os.path.join(self.cache_folder, 'number_of_samples.pt'))


    
    def __getitem__(self, index):
        raise NotImplemented

    def __len__(self):
        raise NotImplemented

class WeimarCollated(WeimarSlicer):

    def __getitem__(self, index):
        return self.X[index] , self.y[index]

    def __len__(self):
        return self.X.size(0)

class WeimarSeparate(WeimarSlicer):

    def __getitem__(self, index):
        start_sample = np.cumsum([0] + self.number_of_samples)
        stop_sample = np.cumsum(self.number_of_samples)
        return (
            self.X[start_sample[index]:stop_sample[index]], 
            self.y[start_sample[index]:stop_sample[index]], 
            self.track_lengths[index]
        )
        

    def __len__(self):
        return len(self.track_lengths)