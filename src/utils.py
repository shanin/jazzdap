import os
import yaml
import IPython.display as ipd

def load_config(filename):
    with open(filename) as f:
        config = yaml.safe_load(f.read())
    return config

def play_audio(sample):
    print(sample['query'])
    return ipd.Audio(rate = sample['sample_rate'], data = sample['audio'])