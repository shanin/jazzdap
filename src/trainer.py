import torch
import numpy as np
import pandas as pd
import mlflow
from tqdm import tqdm

from torch.utils.data import DataLoader

import mir_eval
from utils import weimar2hertz


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
    
    def _parse_config(self, config):
        self.epochs_num = config['crnn_trainer']['epochs_num']
        self.validation_period = config['crnn_trainer']['validation_period']
        self.batch_size = config['crnn_trainer']['batch_size']

    def train_epoch(self):
        self.model.train()

        train_dataloader = DataLoader(
            self.dataset['train-collated'],
            self.batch_size,
            shuffle = True
        )

        for x, y in train_dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            self.optimizer.zero_grad()
            loss = self.criterion(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            mlflow.log_metric('train_loss', loss.item(), self.step_counter)
            self.step_counter += 1    
    
    def calculate_validation_loss(self):
        self.model.eval()
        loss_batches = []

        val_dataloader = DataLoader(
            self.dataset['val-collated'],
            self.batch_size,
            shuffle = False
        )

        with torch.no_grad():
            for x, y in val_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss_batches.append(loss.item())
        mlflow.log_metric('val_loss', np.mean(loss_batches))
        self.model.train()
            
    def train(self):
        self.step_counter = 0
        for epoch_num in tqdm(range(self.epochs_num)):
            self.train_epoch()
            self.calculate_validation_loss()
            self.evaluate('val-separated')
            self.evaluate('test-separated')
            self.scheduler.step()
    
    #probably should be in weimar solo class
    def unfold_predictions(self, pred, track_length):
        if track_length % (pred.size(-2) * pred.size(-3)) != 0:
            unfolded = pred[:-1].reshape(-1, pred.size(-1))
            unfolded_tail = pred[-1].reshape(-1, pred.size(-1))
            tail_len = track_length - unfolded.size(0)
            return torch.cat([unfolded, unfolded_tail[-tail_len:]], dim = 0)
        else:
            return pred.reshape(-1, pred.size(-1))

    def predict(self, X, length):
        self.model.eval()
        test_dataloader = DataLoader(X, self.batch_size, shuffle = False)
        pred_batches = []

        with torch.no_grad():
            for batch in test_dataloader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                pred_batches.append(pred.to('cpu'))

        return self.unfold_predictions(torch.cat(pred_batches, dim = 0), length)


    def evaluate(self, part):
        rows = []
        for x, y, z in self.dataset[part]:
            pitch_estimates = np.argmax(self.predict(x, z).detach().numpy(), axis = 1) + 35
            labels = np.argmax(self.unfold_predictions(y, z).detach().numpy(), axis = 1) + 35

            #some more technical debt
            pitch_estimates[pitch_estimates == 35] = 0
            labels[labels == 35] = 0
    
            evaluation_results = {}

            (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(np.arange(np.size(labels)),
                                                                            weimar2hertz(labels),
                                                                            np.arange(np.size(labels)),
                                                                            weimar2hertz(pitch_estimates))

            vr, vfa = mir_eval.melody.voicing_measures(ref_v, est_v)
            rpa = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)
            rca = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)
            oa = mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)

            evaluation_results['Voicing Recall'] = vr
            evaluation_results['Voicing False Alarm'] = vfa
            evaluation_results['Raw Pitch Accuracy'] = rpa
            evaluation_results['Raw Chroma Accuracy'] = rca
            evaluation_results['Overall Accuracy'] = oa
    
            rows.append(evaluation_results)
        evaluation_results = pd.DataFrame(rows)
        mlflow.log_metric(f'OA_{part}', evaluation_results['Overall Accuracy'].mean())
        mlflow.log_metric(f'VR_{part}', evaluation_results['Voicing Recall'].mean())
        mlflow.log_metric(f'VFA_{part}', evaluation_results['Voicing False Alarm'].mean())
        mlflow.log_metric(f'RPA_{part}', evaluation_results['Raw Pitch Accuracy'].mean())
        mlflow.log_metric(f'RCA_{part}', evaluation_results['Raw Chroma Accuracy'].mean())


            


