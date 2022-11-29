import torch
import numpy as np
import pandas as pd
import mlflow
from tqdm import tqdm
import os

from torch.utils.data import DataLoader

from dataset import PredictedSolo
from scorer import evaluate_sample


class CRNNtrainer:
    def __init__(
        self,
        model = None,
        optimizer = None,
        scheduler = None,
        dataset_dict = None,
        criterion = None,
        device = None,
        config = None
    ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataset = dataset_dict
        self.scheduler = scheduler
        self._parse_config(config)
        self.step_counter = 0
        self.current_val_oa = 0
    
    def _parse_config(self, config):
        self.epochs_num = config['crnn_trainer'].get('epochs_num', 200)
        self.batch_size = config['crnn_trainer'].get('batch_size', 64)
        self.output_folder = os.path.join(
            config['shared'].get('exp_folder', 'exp'),
            config['crnn_trainer'].get('model_folder', 'models')
        )
        self.segment_length = config['crnn']['patch_size'] * config['crnn']['number_of_patches']


    def train_epoch(self):
        self.model.train()

        train_dataloader = DataLoader(
            self.dataset['train-collated'],
            self.batch_size,
            shuffle = True
        )

        for x, y in train_dataloader:
            y = y.type(torch.long)
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(x)
            loss = self.criterion(pred.reshape(-1, 62), y.reshape(-1)) #fix later

            loss.backward()


            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            mlflow.log_metric('train_loss', loss.item(), self.step_counter)
            self.step_counter += 1    
    
    def calculate_validation_loss(self, step = None):
        self.model.eval()
        loss_batches = []

        val_dataloader = DataLoader(
            self.dataset['val-collated'],
            self.batch_size,
            shuffle = False
        )

        with torch.no_grad():
            for x, y in val_dataloader:
                y = y.type(torch.long)
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred.reshape(-1, 62), y.reshape(-1))
                loss_batches.append(loss.item())
        mlflow.log_metric('val_loss', np.mean(loss_batches), step = step)
        self.model.train()
            
    def train(self):
        for epoch_num in tqdm(range(self.epochs_num)):
            mlflow.log_metric("lr", self.optimizer.param_groups[0]["lr"], step=epoch_num)
            self.train_epoch()
            self.calculate_validation_loss(epoch_num)
            val_oa = self.evaluate('val-separated', epoch_num)
            self.evaluate('test-separated', epoch_num)
            self.scheduler.step()
            if val_oa > self.current_val_oa:
                self.current_val_oa = val_oa
                self.save_model()


    def predict(self, X):
        self.model.eval()
        test_dataloader = DataLoader(X, self.batch_size, shuffle = False)
        pred_batches = []

        with torch.no_grad():
            for batch in test_dataloader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                pred_batches.append(pred.to('cpu'))

        return torch.cat(pred_batches, dim = 0)


    def evaluate(self, part, step = None):
        rows = []
        for sf_input, labels, track_length in self.dataset[part]:

            result = PredictedSolo(
                predictions = self.predict(sf_input, track_length),
                labels = labels,
                track_length = track_length,
                segment_length = self.segment_length
            ) 
            
            evaluation_results = evaluate_sample(result.labels, result.predictions)
            rows.append(evaluation_results)
        evaluation_results = pd.DataFrame(rows)

        mlflow.log_metric(f'OA_{part}', evaluation_results['Overall Accuracy'].mean(), step = step)
        mlflow.log_metric(f'VR_{part}', evaluation_results['Voicing Recall'].mean(), step = step)
        mlflow.log_metric(f'VFA_{part}', evaluation_results['Voicing False Alarm'].mean(), step = step)
        mlflow.log_metric(f'RPA_{part}', evaluation_results['Raw Pitch Accuracy'].mean(), step = step)
        mlflow.log_metric(f'RCA_{part}', evaluation_results['Raw Chroma Accuracy'].mean(), step = step)
        return evaluation_results['Overall Accuracy'].mean()

    def save_model(self):
        os.makedirs(self.output_folder, exist_ok=True)
        filename_save_model_st = os.path.join(self.output_folder, f"model_OAval_{self.current_val_oa}.st")
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
        print(f'loaded {filename}')
        self.load_model(os.path.join(self.output_folder, filename))

            


