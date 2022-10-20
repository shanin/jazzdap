from glob import escape
import os
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample
from math import floor
import sqlite3

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

class WeimarDB(Dataset):
    def __init__(self, config):
        config_shared = config['shared']
        config_local = config['dataset']

        self._audio_dir = os.path.join(
            config_shared['raw_data'], 
            config_local['audio_dir']
        )
        self._resample_rate = config_local['resample_rate']
        
        self._init_column_names()
        self._init_database_cursor(config_local)
        self._fetch_data_from_database()
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

    def _init_database_cursor(self, config):
        self._connect = sqlite3.connect(config['weimardb'])
        self._cursor = self._connect.cursor()


    def _fetch_data_from_database(self):
        colnames_str = ', '.join(self._transcription_info_columns)
        query = f'SELECT {colnames_str} FROM transcription_info'
        res = self._cursor.execute(query)
        self._transcription_info = pd.DataFrame(res.fetchall())
        self._transcription_info.columns = self._transcription_info_columns

        colnames_str = ', '.join(self._solo_info_columns)
        query = f'SELECT {colnames_str} FROM solo_info'
        res = self._cursor.execute(query)
        self._solo_info = pd.DataFrame(res.fetchall())
        self._solo_info.columns = self._solo_info_columns

        colnames_str = ', '.join(self._melody_columns)
        query = f'SELECT {colnames_str} FROM melody'
        res = self._cursor.execute(query)
        self._melody = pd.DataFrame(res.fetchall())
        self._melody.columns = self._melody_columns

        colnames_str = ', '.join(self._track_info_columns)
        query = f'SELECT {colnames_str} FROM track_info'
        res = self._cursor.execute(query)
        self._track_info = pd.DataFrame(res.fetchall())
        self._track_info.columns = self._track_info_columns

    def _get_stop_sec(self, start_sec, melody):
        stop_sec = start_sec + melody.onset.max() + 2
        return stop_sec
        

    def __len__(self):
        return len(self._idx_dict)

    def __getitem__(self, idx):
        
        melid = self._idx_dict[idx]
        row = self._transcription_info[self._transcription_info.melid == melid]
        trackid = row.trackid.iloc[0]
        filename = self._track_info[self._track_info.trackid == trackid].filename_track.iloc[0]
        for key in MISTAKES:
            filename = filename.replace(key, MISTAKES[key])
        melody = self._melody[self._melody.melid == melid]
        
        solostart = max(row.solostart_sec.iloc[0], 0) #
        solostop = self._get_stop_sec(solostart, melody)

        data, sample_rate = torchaudio.load(
            os.path.join(
                self._audio_dir, 
                f'{filename}.wav'
            )
        )
        sample_start = floor(solostart * sample_rate)
        if solostop:
            sample_stop = floor(solostop * sample_rate)
        else:
            sample_stop = -1

        solo = data[:, sample_start : sample_stop]

        if self._resample_rate:
            transform = Resample(
                orig_freq = sample_rate, 
                new_freq = self._resample_rate
            )
            solo = transform(solo)
            sample_rate = self._resample_rate
        
        sample = {
            'audio': solo,
            'sample_rate': sample_rate,
            'query': row.filename_solo.iloc[0],
            'melody': melody
        }

        return sample