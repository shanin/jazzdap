import torch
import numpy as np
import pandas as pd
import mlflow
from tqdm import tqdm
import os

from torch.utils.data import DataLoader


class GenericTrainer:
    def __init__(
        self,
        model=None,
        optimizer=None,
        scheduler=None,
        dataset_dict=None,
        criterion=None,
        scorer=None,
        device=None,
        config=None,
    ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataset = dataset_dict
        self.scheduler = scheduler
        self.scorer = scorer
        self._parse_config(config)
        self.step_counter = 0
        self.current_val_oa = 0

    def _parse_config(self, config):
        raise NotImplementedError

    def train_epoch(self):
        raise NotImplementedError

    def calculate_validation_loss(self):
        raise NotImplementedError

    def log_metrics(self, results, part, step):
        raise NotImplementedError

    def train(self):
        for epoch_num in tqdm(range(self.epochs_num)):
            mlflow.log_metric(
                "lr", self.optimizer.param_groups[0]["lr"], step=epoch_num
            )
            self.train_epoch()
            self.calculate_validation_loss(epoch_num)
            eval_validation = self.evaluate("val-separated", epoch_num, log=True)
            self.evaluate("test-separated", epoch_num, log=True)
            self.scheduler.step()
            if eval_validation["Overall Accuracy"].mean() > self.current_val_oa:
                self.current_val_oa = eval_validation["Overall Accuracy"].mean()
                self.save_model()

    def predict(self, X):
        self.model.eval()
        test_dataloader = DataLoader(X, self.batch_size, shuffle=False)
        pred_batches = []

        with torch.no_grad():
            for batch in test_dataloader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                pred_batches.append(pred.to("cpu"))

        return torch.cat(pred_batches, dim=0)

    def evaluate(self, part, step=None, log=False, allow_negative_predictions=True):
        rows = []
        for sf_input, sample in zip(self.dataset[part], self.dataset[part].dataset):

            predictions = self.predict(sf_input)
            sample.generate_predictions_from_net_output(predictions)
            evaluation_results = self.scorer(sample, allow_negative_predictions)
            rows.append(evaluation_results)

        evaluation_results = pd.DataFrame(rows)

        if log:
            self.log_metrics(evaluation_results, part, step)

        return evaluation_results

    def save_model(self):
        os.makedirs(self.output_folder, exist_ok=True)
        filename_save_model_st = os.path.join(
            self.output_folder, f"model_OAval_{self.current_val_oa}.st"
        )
        model = self.model
        model = model.to("cpu")
        torch.save(model.state_dict(), filename_save_model_st)
        mlflow.log_artifact(filename_save_model_st)
        model = model.to(self.device)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()

    def load_best_model(self):
        model_list = sorted(os.listdir(self.output_folder))
        filename = model_list[-1]
        print(f"loaded {filename}")
        self.load_model(os.path.join(self.output_folder, filename))


class CRNNtrainer(GenericTrainer):
    def _parse_config(self, config):
        self.epochs_num = config["crnn_trainer"].get("epochs_num", 200)
        self.batch_size = config["crnn_trainer"].get("batch_size", 64)
        self.output_folder = config["crnn_trainer"].get("model_folder", "models")
        self.segment_length = config["crnn_model"].get("segment_length")

    def train_epoch(self):
        self.model.train()

        train_dataloader = DataLoader(
            self.dataset["train-collated"], self.batch_size, shuffle=True
        )

        for x, y in train_dataloader:
            y = y.type(torch.long)
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(x)
            loss = self.criterion(pred.reshape(-1, 62), y.reshape(-1))  # fix later

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            mlflow.log_metric("train_loss", loss.item(), self.step_counter)
            self.step_counter += 1

    def calculate_validation_loss(self, step=None):
        self.model.eval()
        loss_batches = []

        val_dataloader = DataLoader(
            self.dataset["val-collated"], self.batch_size, shuffle=False
        )

        with torch.no_grad():
            for x, y in val_dataloader:
                y = y.type(torch.long)
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred.reshape(-1, 62), y.reshape(-1))
                loss_batches.append(loss.item())
        mlflow.log_metric("val_loss", np.mean(loss_batches), step=step)
        self.model.train()

    def log_metrics(self, results, part, step):
        mlflow.log_metric(f"OA_{part}", results["Overall Accuracy"].mean(), step=step)
        mlflow.log_metric(f"VR_{part}", results["Voicing Recall"].mean(), step=step)
        mlflow.log_metric(
            f"VFA_{part}", results["Voicing False Alarm"].mean(), step=step
        )
        mlflow.log_metric(
            f"RPA_{part}", results["Raw Pitch Accuracy"].mean(), step=step
        )
        mlflow.log_metric(
            f"RCA_{part}", results["Raw Chroma Accuracy"].mean(), step=step
        )


class OnsetsAndFramesTrainer(GenericTrainer):
    def _parse_config(self, config):
        self.epochs_num = config["onsetsandframes_trainer"].get("epochs_num", 200)
        self.batch_size = config["onsetsandframes_trainer"].get("batch_size", 64)
        self.output_folder = config["onsetsandframes_trainer"].get(
            "model_folder", "exp/onf_models"
        )
        self.segment_length = config["onsetsandframes_model"].get("segment_length", 500)

    def train_epoch(self):
        self.model.train()

        train_dataloader = DataLoader(
            self.dataset["train-collated"], self.batch_size, shuffle=True
        )

        for x, y in train_dataloader:
            # y = y.type(torch.long)
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            onsets_pred, frames_pred = self.model(x)

            # loss = self.criterion(pred.reshape(-1, 62), y.reshape(-1)) #fix later
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
            onsets_loss = self.criterion(onsets_pred, y[:, 0, :, :])
            frames_loss = self.criterion(frames_pred, y[:, 1, :, :]) * y[:, 2, :, :]
            loss = onsets_loss + frames_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            print(onsets_loss.item(), frames_loss.item(), loss.item())
            # mlflow.log_metric('train_loss', loss.item(), self.step_counter)
            self.step_counter += 1

    def calculate_validation_loss(self):
        raise NotImplementedError

    def log_metrics(self, results, part, step):
        raise NotImplementedError
