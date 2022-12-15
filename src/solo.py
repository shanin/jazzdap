import os

import numpy as np
import pandas as pd

import torch
import torchaudio

import sqlite3

import IPython.display as ipd

from wjd_constants import *
from sourcefilter import extract_f0


class Audio:
    """
    A generic Solo class to be used for a fragment of the audio 
    to be transcribed by one of the models in this project.

    ...

    Attributes
    ----------
    audio: torch.FloatTensor
        stereo file, loaded from .wav file of arbitrary sampling rate

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    def _load_audio(self, filename):
        self.filename = filename.split('/')[-1]
        self.audio, self.sample_rate = torchaudio.load(filename)


    def __repr__(self):
        return self.filename

    def play(self):
        print(self.__repr__())
        return ipd.Audio(rate = self.sample_rate, data = self.audio)

    def resampled_audio(self):   
        transform = torchaudio.transforms.Resample(
            orig_freq = self.sample_rate, 
            new_freq = self.resample_rate
        )
        return transform(self.audio)

    def mono_audio(self):
        return self.audio.mean(axis = 0)
        

class GenericMixinFeatures:

    """
    A generic abstract class that for feature generation and 
    saving/loading

    Currently supported features: SfnmfFeatures, CrepeFeatures

    Methods
    -------
    _load_features(filepath: str):
        Loads features from filepath to solo.features attribute.
        Also fills solo.window attribute according to 
        input arguments
    """

    def _generate_features(self):
        raise NotImplementedError

    def _load_features(self, filename):
        raise NotImplementedError

    def _save_features(self, filename):
        raise NotImplementedError


class SfnmfFeatures(GenericMixinFeatures):
    
    def _generate_features(self, Fs = 22050, hop = 256):
        self.window = hop / Fs
        audio_fpath = 'foo'
        input_args = [u'{0}'.format(audio_fpath), 
                      u'--samplingRate={0}'.format(Fs), 
                      u'--hopsize={0}'.format(self.window)]

        _, HF0, *rest = extract_f0(self.mono_audio().numpy(), input_args)
        self.features = HF0.T
        self.num_windows = self.features.shape[0]

    def _load_features(self, filename, Fs = 22050, hop = 256):
        if not filename:
            filename = self.sfnmf_filename
        self.features = np.load(filename).T
        self.window = hop / Fs
        self.num_windows = self.features.shape[0]


class CrepeFeatures(GenericMixinFeatures):

    def _load_features(self, filepath, window = 0.01):
        self.window = window
        self.features = np.load(filepath)
        #cut crepe activations 360 -> 301 (this is ugly so far)
        self.features = self.features[:, 48:]
        self.features = self.features[:, :301]
        self.num_windows = self.features.shape[0]

class GenericMixinLabeling:

    """
    A generic abstract class that for labeling loading and
    transforming to a quantized version

    Currently supported features: CRNNLabeling, OnsetsAndFramesLabeling

    Methods
    -------
    _load_transcription(filepath: str):
        Loads features from filepath to solo.features attribute.
        Also fills solo.window attribute according to 
        input arguments
    """
    
    def _quantized_transcription(self):
        sec2step = lambda x: np.round(x / self.window)
        onsets = self.melody.onset.values
        offsets = self.melody.onset.values + self.melody.duration.values
        pitches = self.melody.pitch.values
        onsets = sec2step(onsets)
        offsets = sec2step(offsets)
        return onsets, offsets, pitches

    def _generate_labels(self):
        raise NotImplementedError


class CRNNLabeling(GenericMixinLabeling):
    def _generate_labels(self):
        lowest_note = 33 # 55Hz, A1, following SF-NMF algorithm
        highest_note = 93 # 1760Hz, A6

        onsets, offsets, pitches = self._quantized_transcription()
        pitch_sequence = np.zeros(self.num_windows)
        for onset, offset, pitch in zip(onsets, offsets, pitches):
            pitch_sequence[int(onset) : int(offset)] = pitch

        pitch_sequence[pitch_sequence > highest_note] = highest_note
        pitch_sequence[pitch_sequence == 0] = lowest_note - 1
        pitch_sequence = pitch_sequence - lowest_note + 1
        pitch_sequence[pitch_sequence < 0] = 1
        self.labels = pitch_sequence
        self.labels_channel_names = ['midi-33']


class OnsetsAndFramesLabeling(GenericMixinLabeling):
    def _generate_labels(self, scaling_const = 5.0, onset_windows = 2):
        
        lowest_note = 33 # 55Hz, A1, following SF-NMF algorithm
        highest_note = 93 # 1760Hz, A6
        weights = 1 + scaling_const / ((np.arange(1000) + 4) / 3)
        onsets, offsets, pitches = self._quantized_transcription()

        raw_onsets = np.zeros((self.num_windows, highest_note - lowest_note + 1))
        raw_frames = np.zeros((self.num_windows, highest_note - lowest_note + 1))
        frame_weights = np.ones((self.num_windows, highest_note - lowest_note + 1))
        for onset, offset, pitch in zip(onsets, offsets, pitches):
            pitch_class = int(min(pitch, highest_note) - lowest_note)
            raw_onsets[int(onset) : int(onset) + onset_windows, pitch_class] = 1
            raw_frames[int(onset): int(offset), pitch_class] = 1
            frame_weights[int(onset) : int(onset) + onset_windows, pitch_class] = scaling_const
            frame_weights[int(onset) + onset_windows: int(offset), pitch_class] = \
                weights[:int(offset) - int(onset) - onset_windows]

        self.labels = np.stack([raw_onsets, raw_frames, frame_weights], axis = 2)
        self.labels_channel_names = ['onsets', 'frames', 'weights']


class GenericMixinPrediction:
    """
    A generic abstract class fro handling predictions generated
    by one of the models. An attribute "predictions" should contain
    a torch.Tensor with the output of network. 

    Currently supported features: CRNNLabeling, OnsetsAndFramesLabeling

    Methods
    -------
    _load_transcription(filepath: str):
        Loads features from filepath to solo.features attribute.
        Also fills solo.window attribute according to 
        input arguments
    """

    def _load_predictions(self, filepath):
        raise NotImplementedError

    def _transform_predictions(self):
        raise NotImplementedError

    def generate_predictions_from_net_output(self, predictions):
        self.predictions = predictions
        self._transform_predictions()

    def save_predictions_to_midi(self, filepath):
        raise NotImplementedError

    def save_predictions_to_csv(self, filepath):
        raise NotImplementedError


class CRNNPrediction(GenericMixinPrediction):

    def _transform_predictions(self):
        self._unfold_predictions()
        self._aggregate_predictions()
        self._class_to_midi()

    def _unfold_predictions(self):
        segment_length = self.predictions.size(-2) * self.predictions.size(-3)
        if self.num_windows % segment_length != 0:
            unfolded = self.predictions[:-1].reshape(-1, self.predictions.size(-1))
            unfolded_tail = self.predictions[-1].reshape(-1, self.predictions.size(-1))
            tail_len = self.num_windows - unfolded.size(0)
            self.predictions = torch.cat([unfolded, unfolded_tail[-tail_len:]], dim = 0)
        else:
            self.predictions = self.predictions.reshape(-1, self.predictions.size(-1))

    def _aggregate_predictions(self):
        self.predictions = self.predictions.detach().numpy()
        voiced_frames = (np.argmax(self.predictions, axis = 1) > 0).astype(int)
        pitch_class = np.argmax(self.predictions[:, 1:], axis = 1) + 1
        pitch_class = pitch_class - 2 * (1 - voiced_frames) * pitch_class
        self.predictions = pitch_class

    def _class_to_midi(self, lowest_note = 33):
        self.predictions[self.predictions > 0] += (lowest_note - 1)
        self.predictions[self.predictions < 0] -= (lowest_note - 1)
        self.labels[self.labels > 0] += (lowest_note - 1)

    def save_predictions_to_csv(self, filepath):
        cleaned_predictions = self.predictions
        cleaned_predictions[cleaned_predictions < 0] = 0
        step = self.window
        prev = 0
        prev_onset = -1
        duration = 0
        notes = []
        for idx, val in enumerate(cleaned_predictions):
            if val != prev:
                if prev != 0:
                    notes.append({
                        'pitch': prev,
                        'duration': (duration + 1) * step,
                        'onset': prev_onset 
                    })
                prev_onset = idx * step 
                duration = 0
                prev = val
            else:
                duration += 1
        pd.DataFrame(notes).to_csv(filepath)

class OnsetAndFramesPrediction(GenericMixinPrediction):
    pass

class MixinInit:
    def __init__(self):
        raise NotImplementedError


class RegularInit(MixinInit):
    def __init__(self, 
                 audio = None, 
                 sample_rate = None,
                 audio_path = None, 
                 features_path = None,
                 transcription_path = None, 
                 output_path = None):
        
        if audio:
            self.audio, self.sample_rate = audio, sample_rate
        elif audio_path:
            self._load_audio(audio_path)
        
        if features_path:
            self._load_features(features_path)
        else:
            self._generate_features()

        if transcription_path:
            self.melody = pd.read_csv(transcription_path)


class WeimarInit(MixinInit):
    
    def _init_database_cursor(self, database_filepath):
        self._connect = sqlite3.connect(database_filepath)
        self._cursor = self._connect.cursor()

    def _parse_transcription_info(self):
        colnames_str = ', '.join(TRANSCRIPTION_INFO_COLNAMES)
        query = f'SELECT {colnames_str} FROM transcription_info WHERE melid = {self.melid}'
        transcription_info = pd.DataFrame(
            self._cursor.execute(query).fetchall(), 
            columns = TRANSCRIPTION_INFO_COLNAMES
        )
        self.trackid = transcription_info.trackid.values[0]
        self.filename_solo = transcription_info.filename_solo.values[0]
        for key in SOLO_MISTAKES:
            self.filename_solo = self.filename_solo.replace(key, SOLO_MISTAKES[key])

    def _parse_solo_info(self):
        colnames_str = ', '.join(SOLO_INFO_COLNAMES)
        query = f'SELECT {colnames_str} FROM solo_info WHERE melid = {self.melid}'
        solo_info = pd.DataFrame(
            self._cursor.execute(query).fetchall(), 
            columns = SOLO_INFO_COLNAMES
        )
        self.instrument = solo_info.instrument.values[0]
        self.performer = solo_info.performer.values[0]
        self.title = solo_info.title.values[0]


    def _parse_track_info(self):
        colnames_str = ', '.join(TRACK_INFO_COLNAMES)
        query = f'SELECT {colnames_str} FROM track_info WHERE trackid = {self.trackid}'
        track_info = pd.DataFrame(
            self._cursor.execute(query).fetchall(), 
            columns = TRACK_INFO_COLNAMES
        )
        filename = track_info.filename_track.values[0]
        for key in MISTAKES:
            filename = filename.replace(key, MISTAKES[key])
        self.filename = filename

    
    def _parse_melody(self):
        colnames_str = ', '.join(MELODY_COLNAMES)
        query = f'SELECT {colnames_str} FROM melody WHERE melid = {self.melid}'
        self.melody = pd.DataFrame(
            self._cursor.execute(query).fetchall(), 
            columns = MELODY_COLNAMES
        )

    def _parse_beats(self):
        colnames_str = ', '.join(BEATS_COLNAMES)
        query = f'SELECT {colnames_str} FROM beats WHERE melid = {self.melid}'
        self.beats = pd.DataFrame(
            self._cursor.execute(query).fetchall(), 
            columns = BEATS_COLNAMES
        )

    
    def __init__(self, melid, database_filepath, audio_folder = None, features_folder = None):
        
        self.features_folder = features_folder
        self.audio_folder = audio_folder
        self.melid = melid

        self._init_database_cursor(database_filepath)
        
        self._parse_transcription_info()
        self._parse_solo_info()
        self._parse_track_info()
        self._parse_melody()
        self._parse_beats()

        if issubclass(self.__class__, Audio):
            filename = os.path.join(
                self.audio_folder,
                f'{self.filename_solo}.wav'
            )
            self._load_audio(filename)

        if issubclass(self.__class__, GenericMixinFeatures):
            feature_filename = os.path.join(
                self.features_folder, 
                f'{str(self.melid).zfill(3)}.npy'
            )
            if os.path.exists(feature_filename):
                self._load_features(feature_filename)
            else:
                self._generate_features()

        if issubclass(self.__class__, GenericMixinLabeling):
            self._generate_labels()

    def __str__(self):
        return(f'{self.performer} ({self.instrument}) - {self.title} (from {self.filename})')
    
    def __repr__(self):
        return(f'{self.performer} ({self.instrument}) - {self.title} (from {self.filename})')

# Several ready classes
class SfnmfCRNNSolo(WeimarInit, SfnmfFeatures, CRNNLabeling, CRNNPrediction):
    pass

# Solo class factory

init_classes = {
    'regular': RegularInit,
    'weimar': WeimarInit
}

raw_data_classes = {
    'audio': Audio
}

feature_classes = {
    'sfnmf': SfnmfFeatures,
    'crepe': CrepeFeatures
}

labeling_classes = {
    'crnn': CRNNLabeling,
    'onsetsandframes': OnsetsAndFramesLabeling
}

prediction_classes = {
    'crnn': CRNNPrediction,
    'onsetsandframes': OnsetAndFramesPrediction
}

def construct_solo_class(dataset_type = 'regular', feature_type = None, 
                         raw_data_type = 'audio', labeling_type = None):
    mixins = []
    mixins.append(init_classes[dataset_type])
    if feature_type:
        mixins.append(feature_classes[feature_type])
    if labeling_type:
        mixins.append(labeling_classes[labeling_type])
        mixins.append(prediction_classes[labeling_type])
    if raw_data_type:
        mixins.append(raw_data_classes[raw_data_type])

    class CustomSolo(*mixins):
        pass

    return CustomSolo