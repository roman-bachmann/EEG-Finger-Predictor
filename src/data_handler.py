import dlc_bci as bci
import torch
from torch.utils.data import TensorDataset, DataLoader


DATA_PATH = '../data_bci'


def load_eeg_data(feature_dim_last=True, standardize=True, one_khz=True):
    """
    Loads the EEG train and test data.

    Args:

        feature_dim_last (boolean, optional): If True, switch the time and feature dimensions

        standardize (boolean, optional): If True, train and test data are standardized

        one_khz (boolean, optional): If True, creates dataset from the 1000Hz data instead
            of the default 100Hz.

    """
    train_input, train_target = bci.load(root=DATA_PATH, train=True, one_khz=one_khz)
    test_input, test_target = bci.load(root=DATA_PATH, train=False, one_khz=one_khz)

    if feature_dim_last:
        train_input = train_input.permute(0,2,1)
        test_input = test_input.permute(0,2,1)

    if standardize:
        train_input, mean, std_dev = standardize_data(train_input)
        test_input, _, _ = standardize_data(test_input, mean, std_dev)

    return train_input, train_target, test_input, test_target


def create_dataloader(train_input, train_target, test_input, test_target, batch_size=64):
    """
    Creates dataloaders for easy mini-batch handling.

    Args:

        train_input (torch.FloatTensor): Input training data

        train_target (torch.FloatTensor): Target training data

        test_input (torch.FloatTensor): Input testing data

        test_target (torch.FloatTensor): Target testing data

        batch_size (int, optional): Size of mini-batches used for training and testing

    """
    train_dataset = TensorDataset(train_input, train_target)
    test_dataset = TensorDataset(test_input, test_target)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    dset_loaders = {'train': trainloader, 'val': testloader}
    dset_sizes = {'train': len(train_input), 'val': len(test_input)}

    return dset_loaders, dset_sizes


def standardize_data(input_data, mean=None, std_dev=None):
    """
    Standardizes the given input data. If mean and standard deviations are given,
    computes the standardisation using those parameters.

    Args:

        input_data (torch.FloatTensor): Input data to standardize

        mean (torch.FloatTensor, optional): Mean along feature axis over all data points

        std_dev (torch.FloatTensor, optional): Standard deviation along feature
            axis over all data points

    """
    if mean is None or std_dev is None:
        mean = input_data.contiguous().view(-1, 28).mean(dim=0)
        std_dev = input_data.contiguous().view(-1, 28).std(dim=0)
    return (input_data - mean) / std_dev, mean, std_dev


def augment_dataset(train_input, train_target, std_dev, multiple):
    """
    Augments the size of the dataset by introducing unbiased gaussian noise.
    Resulting dataset is 'multiple' times bigger than original.

    Args:

        train_input (torch.FloatTensor): Input training data

        train_target (torch.FloatTensor): Target training data

        std_dev (float): Standard deviation of gaussian noise to apply

        multiple (int): Factor by how much the dataset should be bigger
    """
    new_train_input = train_input.clone()
    new_train_target = train_target.clone()
    for i in range(multiple-1):
        augmented_input = torch.zeros_like(train_input).normal_(0, std_dev)
        #augmented_input = train_input + torch.zeros(train_input.shape).normal_(0, std_dev)
        new_train_input = torch.cat((new_train_input, augmented_input))
        new_train_target = torch.cat((new_train_target, train_target))
    return new_train_input, new_train_target
