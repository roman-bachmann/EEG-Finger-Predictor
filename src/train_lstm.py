import torch
from data_handler import load_eeg_data, create_dataloader, augment_dataset
from models import LSTM_Model
from train_utils import k_fold_cv, train_model
from plots import plot_learning_curves, plot_CV_learning_curves

batch_size=64
std_dev=0.1
multiple=2

# Loading and processing the data
train_input, train_target, test_input, test_target = load_eeg_data()
dset_loaders, dset_sizes = create_dataloader(train_input, train_target, \
                                             test_input, test_target, \
                                             batch_size=batch_size)
aug_train_input, aug_train_target = augment_dataset(train_input, train_target, std_dev, multiple)

# Defining the model
model = LSTM_Model(train_input.shape[2], hidden_size=128, num_layers=1, dropout=0.1)
criterion = torch.nn.CrossEntropyLoss()
learning_rate = 1e-4
weight_decay = 1e-3 # L2 regularizer parameter
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

if torch.cuda.is_available():
    model.cuda()
    criterion.cuda()

# Training and plotting the model
best_model, train_losses, val_losses, train_accs, val_accs = train_model(model, dset_loaders, \
                                                                         dset_sizes, criterion, \
                                                                         optimizer, num_epochs=5, \
                                                                         verbose=2)

plot_learning_curves(train_losses, val_losses, train_accs, val_accs)
