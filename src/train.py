import argparse

import numpy as np
import torch

from utils import load_config

from dataset import WeimarDB

from sampler import (
    CRNNSamplerInference,
    CRNNSamplerTraining,
    OnsetsAndFramesSamplerInference,
    OnsetsAndFramesSamplerTraining,
)
from model import CRNN, OnsetsAndFrames
from trainer import CRNNtrainer, OnsetsAndFramesTrainer


from scorer import evaluate_sample

import mlflow


def prepare_dataset(
    config,
    partition,
    sampler_type,
    feature_type,
    data_tag,
    inference_sampler,
    training_sampler,
):

    if sampler_type == "inference":
        sampler = inference_sampler
    else:
        sampler = training_sampler

    test_time = sampler_type == "inference"

    dataset = sampler(
        WeimarDB(config, partition=partition),
        config,
        tag=f"{partition}-{feature_type}-{data_tag}",
        test_time=test_time,
    )

    return dataset


def setup_dataset_dict(
    config, parts, types, feature_type, inference_sampler, training_class
):
    data_tag = config.get("data_tag", "default")
    dataset_dict = {}
    for part, type in zip(parts, types):
        print(f"loading partition: {part}-{feature_type}-{type}")
        dataset_dict[f"{part}-{type}"] = prepare_dataset(
            config,
            part,
            type,
            feature_type,
            data_tag,
            inference_sampler,
            training_class,
        )
    return dataset_dict


def setup_device(config):
    index = config["device"]
    return torch.device(f"cuda:{index}" if torch.cuda.is_available() else "cpu")


def setup_optimizer(config, parameters):
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0)
    mlflow.log_param("weight_decay", weight_decay)
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def setup_scheduler(config, optimizer):
    lr = config["lr"]
    lr_final = config["lr_final"]
    epochs_num = config["epochs_num"]
    gamma = np.power(lr_final / lr, 1 / epochs_num)
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


def setup_criterion(config):
    label_smoothing = config.get("label_smoothing", 0)
    mlflow.log_param("label_smoothing", label_smoothing)
    return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def setup_trainer(
    config,
    model=None,
    trainer_class=CRNNtrainer,
    inference_sampler=CRNNSamplerInference,
    training_class=CRNNSamplerInference,
    partitions=["train", "test", "val", "val"],
    modes=["train", "inference", "train", "inference"],
    feature_type="sfnmf",
    test_time_dataset=None,
):

    parameters = model.parameters()
    if test_time_dataset:
        dataset_dict = test_time_dataset
    else:
        dataset_dict = setup_dataset_dict(
            config, partitions, modes, feature_type, inference_sampler, training_class
        )
    device = setup_device(config)
    criterion = setup_criterion(config)
    optimizer = setup_optimizer(config, parameters)
    scheduler = setup_scheduler(config, optimizer)

    trainer = trainer_class(
        device=device,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataset_dict=dataset_dict,
        scorer=evaluate_sample,
        scheduler=scheduler,
        config=config,
    )

    return trainer


def setup_crnn_trainer(config):
    return setup_trainer(
        config["crnn_trainer"],
        CRNN(config),
        CRNNtrainer,
        CRNNSamplerInference,
        CRNNSamplerTraining,
    )

def setup_onf_trainer(config):
    return setup_trainer(
        config["onsetsandframes_trainer"],
        OnsetsAndFrames(config),
        OnsetsAndFramesTrainer,
        OnsetsAndFramesSamplerInference,
        OnsetsAndFramesSamplerTraining,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logging-level", type=str, default="WARNING")
    args = parser.parse_args()

    config = load_config(args.config)

    mlflow.start_run()

    trainer = setup_crnn_trainer(config)
    trainer.train()

    mlflow.end_run()
