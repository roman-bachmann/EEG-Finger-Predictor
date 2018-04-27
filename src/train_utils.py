import torch
from torch.autograd import Variable
import numpy as np
import time
import copy


def train_model(model, dset_loaders, dset_sizes, criterion, optimizer, \
                lr_scheduler=None, num_epochs=25, verbose=2):
    """
    Method to train a PyTorch neural network with the given parameters for a
    certain number of epochs. Keeps track of the model yielding the best validation
    accuracy during training and returns that model before potential overfitting
    starts happening.
    Records and returns training and validation losses and accuracies over all
    epochs.

    Args:

        model (torch.nn.Module): The neural network model that should be trained.

        dset_loaders (dict(string, DataLoader)): Dictionary containing the training
            loader and test loader: {'train': trainloader, 'val': testloader}

        dset_sizes (dict(string, int)): Dictionary containing the size of the training
            and testing sets. {'train': train_set_size, 'val': test_set_size}

        criterion (PyTorch criterion): PyTorch criterion (e.g. CrossEntropyLoss)

        optimizer (PyTorch optimizer): PyTorch optimizer (e.g. Adam)

        lr_scheduler (PyTorch learning rate scheduler, optional): PyTorch learning rate scheduler

        num_epochs (int): Number of epochs to train for

        verbose (int): Verbosity level. 0 for none, 1 for small and 2 for heavy printouts
    """
    start_time = time.time()

    best_model = model
    best_acc = 0.0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        if verbose > 1:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if lr_scheduler:
                    optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # zero the hidden state
                hidden = model.init_hidden(len(inputs))

                # forward
                outputs = model(inputs, hidden)
                #_, preds = torch.max(outputs.data, 1)
                preds = outputs
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                preds_classes = preds.data.max(1)[1]
                running_corrects += torch.sum(preds_classes == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

            if verbose > 1:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        if verbose > 1:
            print()

    time_elapsed = time.time() - start_time
    if verbose > 0:
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    return best_model, train_losses, val_losses, train_accs, val_accs


def k_fold_cv(model, train_input, train_target, criterion, optimizer, lr_scheduler=None,\
              num_epochs=25, batch_size=64, K=5, verbose=2, augment_multiplier=0, std_dev=0.1):
    """
    Method to run K-fold Cross Validation on a PyTorch neural network with the
    given parameters for a certain number of epochs.
    Records and returns training and validation losses and accuracies over all
    epochs over all K folds.
    If training dataset size is augmented, it will for each fold split the training
    set into training and validation set and then augment the new training set only.
    Like this, the validation set is independent of the augmented training data
    for every fold. All different folds will therefore have different random noise
    added to it.

    Args:

        model (torch.nn.Module): The neural network model that should be trained.

        train_input (): Training input data

        train_target (): Training target data

        criterion (PyTorch criterion): PyTorch criterion (e.g. CrossEntropyLoss)

        optimizer (PyTorch optimizer): PyTorch optimizer (e.g. Adam)

        lr_scheduler (PyTorch learning rate scheduler, optional): PyTorch learning rate scheduler

        num_epochs (int): Number of epochs to train for

        batch_size (int): Size of mini-batches

        verbose (int): Verbosity level. 0 for none, 1 for small and 2 for heavy printouts

        augment_multiplier (int): Factor by how much the dataset should be bigger

        std_dev (float): Standard deviation of gaussian noise to apply
    """
    n_train = len(train_input)
    indices = list(range(n_train))
    n_validation = n_train // K

    np.random.shuffle(indices)

    avg_train_loss, avg_val_loss = 0, 0
    avg_train_acc, avg_val_acc = 0, 0

    K_train_losses = []
    K_val_losses = []
    K_train_accs = []
    K_val_accs = []

    for k in range(K):
        indices_rolled = np.roll(indices, k * n_train // K)
        train_idx, val_idx = indices_rolled[n_validation:], indices_rolled[:n_validation]

        train_inp = train_input[train_idx,]
        train_tar = train_target[train_idx,]
        val_inp = train_input[val_idx,]
        val_tar = train_target[val_idx,]

        train_inp, train_tar = augment_dataset(train_inp, train_tar, std_dev, augment_multiplier)

        train_dataset = TensorDataset(train_inp, train_tar)
        val_dataset = TensorDataset(val_inp, val_tar)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        dset_loaders = {'train': train_loader, 'val': val_loader}
        dset_sizes = {'train': len(train_inp), 'val': len(val_inp)}

        if verbose:
            print('k={}:'.format(k))
        _, train_losses, val_losses, train_accs, val_accs = train_model(copy.deepcopy(model), dset_loaders,
                                                                        dset_sizes, criterion, optimizer,
                                                                        num_epochs=num_epochs, verbose=verbose)

        avg_train_loss += min(train_losses)
        avg_val_loss += min(val_losses)
        avg_train_acc += max(train_accs)
        avg_val_acc += max(val_accs)

        K_train_losses.append(train_losses)
        K_val_losses.append(val_losses)
        K_train_accs.append(train_accs)
        K_val_accs.append(val_accs)

    avg_train_loss /= K
    avg_val_loss /= K
    avg_train_acc /= K
    avg_val_acc /= K

    if verbose:
        print('\nAvg best train loss: {:.6f}, avg best val loss: {:.6f}'.format(avg_train_loss, avg_val_loss))
        print('Avg best train acc: {:.6f}%, avg best val acc: {:.6f}%'.format(avg_train_acc*100, avg_val_acc*100))

    return K_train_losses, K_val_losses, K_train_accs, K_val_accs
