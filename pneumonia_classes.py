import random
import csv
from pathlib import Path
from statistics import mean
from copy import copy
from time import time

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class PneumoniaDataset(Dataset):
    def __init__(self, data_path, img_size, debug_size):
        """

        :param data_path:
        :param img_size: Defaults to appropriate value for this dataset
        (check results from running img_dims.py)
        :param debug_size:
        """
        self.img_size = img_size
        if debug_size:
            self.data_path = tuple(
                [file for file in Path(data_path).glob('*')][:48])
        else:
            self.data_path = tuple([file for file in Path(data_path).glob('*')])
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        img_path = self.data_path[idx]
        if "normal" in img_path.name:
            label = 0
        elif "pneumonia" in img_path.name:
            label = 1
        else:
            raise Exception(f'{img_path.name} cannot be classified.')
        img = Image.open(img_path)
        return self._get_image_transformation()(img), label

    def _get_image_transformation(self):
        transform_list = []
        transform_list.append(transforms.Grayscale())
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomVerticalFlip())
        rotation_angle = random.uniform(-30, 30)
        transform_list.append(transforms.RandomRotation(
            degrees=(rotation_angle, rotation_angle)))
        transform_list.append(transforms.Resize(self.img_size))
        transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)


class PneumoniaClassifier():
    def __init__(self, model:torch.nn.Module,
                 train_data: DataLoader,
                 val_data: DataLoader,
                 num_epochs: int,
                 learning_rate: float,
                 patience: int):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_info = str(self.model)
        self.train_data = train_data
        self.val_data = val_data
        self.num_epochs = num_epochs
        self.patience = patience
        self.loss_function = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate)
        self.classifier_has_been_trained = False
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_accuracy = 0
        self.best_weights = None
        print(self.model_info + 2 * '\n')

    def _compare_and_save_performance(self, val_accuracy):
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_weights = copy(self.model.state_dict())

    def _compute_accuracies(self, pred, y):
        """Compute the classification accuracy for a set of predictions."""
        return float((pred.round() == y).float().mean())

    def _train_loop(self):
        """Perform the training step of an epoch."""
        self.model.train()
        batch_train_losses = []
        batch_train_accuracies = []
        for batch, (X, y) in enumerate(self.train_data):
            y = torch.reshape(y, (y.size(dim=0), 1))
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = self.loss_function(pred, y.float())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            batch_train_losses.append(loss.item())
            batch_train_accuracies.append(self._compute_accuracies(pred, y))

        epoch_train_loss = mean(batch_train_losses)
        epoch_train_accuracy = mean(batch_train_accuracies)
        self.train_losses.append(epoch_train_loss)
        self.train_accuracies.append(epoch_train_accuracy)
        print(f'train loss: {round(epoch_train_loss, 3)}')
        print(f'train acc: {round(epoch_train_accuracy, 3)}')

    def _eval_loop(self):
        self.model.eval()
        batch_val_losses = []
        batch_val_accuracies = []
        with torch.no_grad():
            for X, y in self.val_data:
                y = torch.reshape(y, (y.size(dim=0), 1))
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.loss_function(pred, y.float())
                batch_val_losses.append(loss.item())
                batch_val_accuracies.append(self._compute_accuracies(pred, y))
        epoch_val_loss = mean(batch_val_losses)
        epoch_val_accuracy = mean(batch_val_accuracies)
        self.val_losses.append(epoch_val_loss)
        self.val_accuracies.append(epoch_val_accuracy)
        print(f'val loss: {round(epoch_val_loss, 3)}')
        print(f'val acc: {round(epoch_val_accuracy, 3)}')
        self._compare_and_save_performance(epoch_val_accuracy)

    def train_network(self):
        for e in range(1, self.num_epochs+1):
            print(f'------- Epoch {e} -------')
            self._train_loop()
            self._eval_loop()

        self.classifier_has_been_trained = True

    def plot_and_save_losses(self, plot_name):
        if not self.classifier_has_been_trained:
            print('Classifier has not been trained yet.')
        else:
            epochs = range(1, self.num_epochs+1)
            plt.figure(figsize=(16, 12))
            plt.plot(epochs,
                     self.train_losses,
                     label='Training Loss',
                     marker='o',
                     linestyle='-')
            plt.plot(epochs,
                     self.val_losses,
                     label='Validation Loss',
                     marker='o',
                     linestyle='-')
            plt.plot(epochs,
                     self.train_accuracies,
                     label='Training Accuracy',
                     marker='o',
                     linestyle='-')
            plt.plot(epochs,
                     self.val_accuracies,
                     label='Validation Accuracy',
                     marker='o',
                     linestyle='-')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Train and val loss.')
            plt.legend()
            plt.grid(True)
            plt.savefig(plot_name)
            plt.show()

    def get_best_accuracy_and_model(self):
        return (self.model_info,
                round(self.best_accuracy, 3),
                self.best_weights)


class PerformanceWriter():
    def __init__(self, performance_table_path: str,
                 model_state_dict_directory: str,
                 model_state_dict_file: torch.nn.Module.state_dict,
                 model_info: str,
                 hyperparameters: dict,
                 accuracy: float):
        self.performance_table_path = Path(performance_table_path)
        self.model_state_dict_directory = Path(model_state_dict_directory)
        self.model_state_dict_file = model_state_dict_file
        self.write_content = hyperparameters
        self.write_content['accuracy'] = accuracy
        self.write_content['state_dict_name'] = (
            str(time()).replace('.', ''))
        self.write_content['model_info'] = model_info.replace('\n', '')
        self.column_names = list(hyperparameters.keys())
        if not self.performance_table_path.is_file():
            with open(self.performance_table_path, 'w') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(self.column_names)
        if not  self.model_state_dict_directory.is_dir():
            Path.mkdir(self.model_state_dict_directory)

    def write_accuracy_and_hyperparameters_and_state_dict(self):
        with open(self.performance_table_path, 'a') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=self.column_names,
                                    delimiter=';')
            writer.writerow(self.write_content)
        file_path = Path(self.model_state_dict_directory,
                         self.write_content['state_dict_name'] + '.pth')
        torch.save(self.model_state_dict_file, file_path)

