import torch
import itertools
from data_handler import load_eeg_data, create_dataloader, augment_dataset
from models import CNN_Model
from train_utils import k_fold_cv, train_model
from plots import plot_learning_curves, plot_CV_learning_curves

if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn
    cudnn.enabled = True
    cudnn.benchmark = True


# Loading and processing the data
print('Loading the data')
train_input, train_target, _, _ = load_eeg_data(feature_dim_last=True,
                                                standardize=True, one_khz=True)

learning_rate = 1e-4
weight_decay = 1e-3 # L2 regularizer parameter

params = {'conv_channels': [[28, 32 ,1], [28, 64, 1], [28, 32, 16, 1], [28, 64, 32, 1]],
          'augment_multiplier': [1,5,15],
          'std_dev': [0.01, 0.1]}

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

keys = list(params)
for values in itertools.product(*map(params.get, keys)):
    d = dict(zip(keys, values))
    description = 'C{}M{}STD{}'.format(d['conv_channels'], d['augment_multiplier'], d['std_dev'])

    # Skip duplicate training
    if d['augment_multiplier'] == 1 and d['std_dev'] == 0.1:
        continue

    print('\n\n##### ' + description + ' #####')

    # Defining the model
    model = CNN_Model(train_input.shape[1], kernel_sizes=[3,5,7],
                      conv_channels=d['conv_channels'], dropout=0)
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    cv_curves = k_fold_cv(model, train_input, train_target,
                          criterion, learning_rate, weight_decay,
                          num_epochs=25, batch_size=64, K=5, verbose=1,
                          augment_multiplier=d['augment_multiplier'], std_dev=d['std_dev'])

    plot_CV_learning_curves(cv_curves[0], cv_curves[1], cv_curves[2], cv_curves[3], description)
