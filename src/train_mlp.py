import torch
from data_handler import load_eeg_data
from train_utils import k_fold_cv, train_model
from plots import plot_learning_curves, plot_CV_learning_curves


# Loading and processing the data
print('Loading the data')
train_input, train_target, _, _ = load_eeg_data(feature_dim_last=True,
                                                standardize=True, one_khz=True)
train_input = train_input.view(train_input.shape[0], -1)

# Defining the model
D_in = train_input.shape[1]
H = 256
D_out = 2

model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        )

criterion = torch.nn.CrossEntropyLoss()
learning_rate = 1e-4
weight_decay = 1e-3 # L2 regularizer parameter

if torch.cuda.is_available():
    model.cuda()
    criterion.cuda()

cv_curves = k_fold_cv(model, train_input, train_target,
                      criterion, learning_rate, weight_decay,
                      num_epochs=25, batch_size=64, K=5, verbose=1)

plot_CV_learning_curves(cv_curves[0], cv_curves[1], cv_curves[2], cv_curves[3], 'mlp')
