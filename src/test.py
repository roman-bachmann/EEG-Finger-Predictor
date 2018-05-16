import torch
from torch.autograd import Variable
from data_handler import load_eeg_data, create_dataloader, augment_dataset
from models import CNN_Model
torch.manual_seed(12345) # For reproducibility

# Set cudnn flags to True for speed increase, but lose reproducibility
if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn
    cudnn.enabled = False
    cudnn.benchmark = False

# Loading and processing the data
print('Loading the data')
train_input, train_target, test_input, test_target = load_eeg_data(feature_dim_last=True,
                                                                   standardize=True, one_khz=True)

train_input, train_target = augment_dataset(train_input, train_target, 0.1, 15)
dset_loaders, dset_sizes = create_dataloader(train_input, train_target, test_input, test_target, batch_size=64)


# Defining the model
model = CNN_Model(train_input.shape[1], kernel_sizes=[3,5,7],
                  conv_channels=[28,32,16,1], dropout=0.1)
criterion = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3
weight_decay = 1e-4 # L2 regularizer parameter
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

if torch.cuda.is_available():
    model.cuda()
    criterion.cuda()


# Training the model
model.train(True)

num_epochs=6
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)

    running_loss = 0.0
    running_corrects = 0

    # Iterate over mini-batches
    for data in dset_loaders['train']:
        inputs, labels = data

        # Wrap inputs and labels in Variable
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), \
                Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs)

        loss = criterion(preds, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        preds_classes = preds.data.max(1)[1]
        running_corrects += torch.sum(preds_classes == labels.data)

    epoch_loss = running_loss / dset_sizes['train']
    epoch_acc = running_corrects / dset_sizes['train']

    print('Train loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))


# Evaluating final model on test data
model.train(False)

running_loss = 0.0
running_corrects = 0

# Iterate over mini-batches
for data in dset_loaders['val']:
    inputs, labels = data

    # Wrap inputs and labels in Variable
    if torch.cuda.is_available():
        inputs, labels = Variable(inputs.cuda()), \
            Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    preds = model(inputs)

    loss = criterion(preds, labels)
    running_loss += loss.data[0]
    preds_classes = preds.data.max(1)[1]
    running_corrects += torch.sum(preds_classes == labels.data)

epoch_loss = running_loss / dset_sizes['val']
epoch_acc = running_corrects / dset_sizes['val']

print('Final test loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))
