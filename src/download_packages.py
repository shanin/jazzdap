import os
import argparse
from utils import load_config

def download(config):
    folder_name = config['package_dir']
    package_dict = config['packages']
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for key in package_dict:
        os.system(f'git clone {package_dict[key]} {folder_name}/{key}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logging-level", type=str, default="WARNING")
    args = parser.parse_args()

    config = load_config(args.config)
    download(config['download_packages'])