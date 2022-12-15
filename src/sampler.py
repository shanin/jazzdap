import os
import numpy as np
from torch.utils.data import Dataset
import torch

from math import floor

from sklearn.preprocessing import normalize
from tqdm import tqdm

from wjd_constants import *

class SoloSampler(Dataset):

    def _parse_config(self, config):
        raise NotImplementedError

    def _cut(self, sample, features, labels):
        first_onset = sample.melody.onset.iloc[0]
        last_offset = sample.melody.onset.iloc[-1] + sample.melody.duration.iloc[-1]
        start = floor(first_onset * (self.sampling_rate / self.hop))
        stop = floor(last_offset * (self.sampling_rate / self.hop)) + 1
        return features[start:stop], labels[start:stop]

    def _prepare_tensors_from_sample(self, features, labels):
        raise NotImplementedError

    def _assemble_tensors(self):

        X_list_of_tensors = []
        y_list_of_tensors = []
        track_lengths = []
        number_of_samples = []

        for sample in tqdm(self.dataset):
            features = sample.features
            labels = sample.labels if hasattr(sample, 'labels') else []

            if not self.test_time:
                features, labels = self._cut(sample, features, labels)
            
            HF0_tensor, hf0_len, samples = self._prepare_HF0_tensor(features)
            labels_tensor, _, _ = self._prepare_labels_tensor(labels)

            X_list_of_tensors.append(HF0_tensor)
            y_list_of_tensors.append(labels_tensor)
            track_lengths.append(hf0_len)
            number_of_samples.append(samples)

        self.X = torch.cat(X_list_of_tensors, dim = 0)
        self.y = torch.cat(y_list_of_tensors, dim = 0)
        self.track_lengths = track_lengths
        self.number_of_samples = number_of_samples


    def __init__(self, dataset, config, tag = None, test_time = False):

        # refactor: the problem here is that y.pt is not created for SeparateSampler

        self._parse_config(config)
        self.tag = tag
        self.dataset = dataset
        self.test_time = test_time

        if self.tag is None:
            self._assemble_tensors()
        else:
            #use cached version (or save generated version to cache)
            self.cache_folder = os.path.join(
                config['sampler']['cache_folder'],
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

class GenericFrameLevelSampler(SoloSampler):

    def _prepare_labels_tensor(self, labels):
        raise NotImplementedError

    def _prepare_tensors_from_sample(self, features, labels):
        HF0, length_of_sequence, number_of_samples = self._prepare_HF0_tensor(features)
        labels, _, _ = self._prepare_labels_tensor(labels)
        tensors = {
            'HF0': HF0,
            'labels': labels
        }
        metadata = {
            'length_of_sequence': length_of_sequence,
            'number_of_samples': number_of_samples
        }
        return tensors, metadata

class GenericCRNNSampler(GenericFrameLevelSampler):

    def _parse_config(self, config):
        self.segment_length = config['crnn_model']['segment_length']
        self.feature_size = config['crnn_model']['feature_size']
        self.number_of_classes = config['crnn_model']['number_of_classes']
        self.features_type = config['weimar_dataset']['feature_type']
        self.sampling_rate = config[f'{self.features_type}_features']['Fs']
        self.hop = config[f'{self.features_type}_features']['hop']
        self.number_of_patches = config['crnn_model']['number_of_patches']
        self.patch_size = config['crnn_model']['patch_size']
        self.num_label_channels = 1

    def _prepare_HF0_tensor(self, features):
        length_of_sequence = features.shape[0]
        number_of_segments = int(floor(length_of_sequence/self.segment_length))

        HF0 = np.append(
            features[: number_of_segments * self.segment_length],
            features[-self.segment_length: ], 
            axis=0
        )
        HF0 = normalize(HF0, norm='l1', axis=1)

        number_of_samples = int(HF0.shape[0] / self.segment_length)
        HF0 = np.reshape(
            HF0, 
            (number_of_samples, 1, self.number_of_patches, self.patch_size, self.feature_size)
        )
        return torch.tensor(HF0, dtype=torch.float), length_of_sequence, number_of_samples

    def _prepare_labels_tensor(self, labels):
        if len(labels):
            length_of_sequence = labels.shape[0]
            number_of_segments = int(floor(length_of_sequence/self.segment_length))

            y = np.append(
                labels[:(number_of_segments * self.segment_length)],
                labels[-self.segment_length: ], 
                axis=0
            )

            number_of_samples = int(y.shape[0] / (self.segment_length))
            y = np.reshape(
                y,
                (number_of_samples, self.number_of_patches, self.patch_size, self.num_label_channels)
            )
            return torch.tensor(y, dtype=torch.int).transpose(1,3).transpose(2,3), length_of_sequence, number_of_samples
        else:
            return torch.empty(0), None, None

class CRNNSamplerTraining(GenericCRNNSampler):

    def __getitem__(self, index):
        return self.X[index] , self.y[index]

    def __len__(self):
        return self.X.size(0)

class CRNNSamplerInference(GenericCRNNSampler):

    def __getitem__(self, index):
        start_sample = np.cumsum([0] + self.number_of_samples)
        stop_sample = np.cumsum(self.number_of_samples)
        return self.X[start_sample[index]:stop_sample[index]]
        
    def __len__(self):
        return len(self.track_lengths)


class GenericOnsetsAndFramesSampler(GenericFrameLevelSampler):

    def _parse_config(self, config):
        self.segment_length = config['onsetsandframes_model']['segment_length']
        self.feature_size = config['onsetsandframes_model']['feature_size']
        self.number_of_classes = config['onsetsandframes_model']['number_of_classes']
        self.num_label_channels = config['onsetsandframes_model']['num_label_channels']
        self.features_type = config['weimar_dataset']['feature_type']
        self.sampling_rate = config[f'{self.features_type}_features']['Fs']
        self.hop = config[f'{self.features_type}_features']['hop']

    def _prepare_HF0_tensor(self, features):
        length_of_sequence = features.shape[0]
        number_of_segments = int(floor(length_of_sequence/self.segment_length))

        HF0 = np.append(
            features[: number_of_segments * self.segment_length],
            features[-self.segment_length: ], 
            axis=0
        )
        HF0 = normalize(HF0, norm='l1', axis=1)

        number_of_samples = int(HF0.shape[0] / self.segment_length)
        HF0 = np.reshape(
            HF0, 
            (number_of_samples, 1, self.segment_length, self.feature_size)
        )
        return torch.tensor(HF0, dtype=torch.float), length_of_sequence, number_of_samples

    def _prepare_labels_tensor(self, labels):
        if len(labels):
            length_of_sequence = labels.shape[0]
            number_of_segments = int(floor(length_of_sequence/self.segment_length))

            y = np.append(
                labels[:(number_of_segments * self.segment_length)],
                labels[-self.segment_length: ], 
                axis=0
            )

            number_of_samples = int(y.shape[0] / (self.segment_length))
            y = np.reshape(
                y,
                (number_of_samples, self.segment_length, self.number_of_classes, self.num_label_channels)
            )
            return torch.tensor(y, dtype=torch.float).transpose(1,3).transpose(2,3), length_of_sequence, number_of_samples
        else:
            return torch.empty(0), None, None

#maybe this should be a mix-in
class OnsetsAndFramesSamplerTraining(GenericOnsetsAndFramesSampler):

    def __getitem__(self, index):
        return self.X[index] , self.y[index]

    def __len__(self):
        return self.X.size(0)

class OnsetsAndFramesSamplerInference(GenericOnsetsAndFramesSampler):

    def __getitem__(self, index):
        start_sample = np.cumsum([0] + self.number_of_samples)
        stop_sample = np.cumsum(self.number_of_samples)
        return self.X[start_sample[index]:stop_sample[index]]
        
    def __len__(self):
        return len(self.track_lengths)
