import random
import datetime
from pathlib import Path
from statistics import mean

import torch
from PIL import Image

import torch
from torch.utils.data import Dataset
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
    def __init__(self, model, train_data, val_data,
                 num_epochs, learning_rate):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        print(str(self.model) + 2 * '\n')
        self.train_data = train_data
        self.val_data = val_data
        self.num_epochs = num_epochs
        self.loss_function = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate)
        self.classifier_has_been_trained = False
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def _get_accuracies(self, pred, y):
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
            batch_train_accuracies.append(self._get_accuracies(pred, y))

        epoch_train_loss = mean(batch_train_losses)
        epoch_train_accuracy = mean(batch_train_accuracies)
        self.train_losses.append(epoch_train_loss)
        self.train_accuracies.append(epoch_train_accuracy)
        print(f'train loss: {round(epoch_train_loss, 3)}')
        print(f'train acc: {round(epoch_train_accuracy, 3)}')


    # eval loop
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
                batch_val_accuracies.append(self._get_accuracies(pred, y))
        epoch_val_loss = mean(batch_val_losses)
        epoch_val_accuracy = mean(batch_val_accuracies)
        self.val_losses.append(epoch_val_loss)
        self.val_accuracies.append(epoch_val_accuracy)
        print(f'val loss: {round(epoch_val_loss, 3)}')
        print(f'val acc: {round(epoch_val_accuracy, 3)}')

        # calculate prop of correct etc

    def train_network(self):
        for e in range(1, self.num_epochs+1):
            print(f'------- Epoch {e} -------')
            self._train_loop()
            self._eval_loop()

        self.classifier_has_been_trained = True

    # method to do the testing

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

