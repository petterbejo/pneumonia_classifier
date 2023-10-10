"""
Run the classifier.
"""
from datetime import datetime

from torch.utils.data import random_split, DataLoader

from cnn_architectures import VeryBasicNN
from pneumonia_classes import PneumoniaDataset, PneumoniaClassifier

data_path = 'data/data'
img_size = (768, 768)
num_epochs = 2
learning_rate = 1e-3
debug_size = True
batch_size = 32
patience = 5
plot_time = str(datetime.now().year)[2:] + str(datetime.now().month) + \
            str(datetime.now().day) + '-' + str(datetime.now().hour) + \
            str(datetime.now().minute)
plot_dir = 'plots'
plot_name = (plot_dir + '/debug_size_' + str(debug_size).lower() + '_epochs_' +
             str(num_epochs) + '_lr_' + str(learning_rate) + '_img_size_' +
             str(img_size) + plot_time + '.png')
best_model_dir = 'best_model'
hyperparameters = {'img_size': img_size,
                  'num_epochs': num_epochs,
                  'batch_size': batch_size,
                  'early_stopping_patience': patience,
                  'learning_rate': learning_rate}

dataset = PneumoniaDataset(data_path,
                           img_size=img_size,
                           debug_size=debug_size)
train, val, test = random_split(dataset, [0.7, 0.2, 0.1])
train_loader = DataLoader(dataset=train,
                          batch_size=batch_size,
                          shuffle=True)
val_loader = DataLoader(dataset=val,
                        batch_size=batch_size,
                        shuffle=True)
test_loader = DataLoader(dataset=test,
                         batch_size=batch_size,
                         shuffle=True)

model = VeryBasicNN(img_size=img_size)
classifier = PneumoniaClassifier(model=model,
                                 train_data=train_loader,
                                 val_data=val_loader,
                                 num_epochs=num_epochs,
                                 learning_rate=learning_rate,
                                 patience=patience)
classifier.train_network()
classifier.plot_and_save_losses(plot_name=plot_name)


