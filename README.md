# EEG-Finger-Predictor
First project of the EPFL Spring 2018 Deep Learning Class

## Dependencies

- Python 3.5.4
- PyTorch 3.0.1
- Numpy 1.13.3
- Matplotlib 2.0.2

## Folder structure

- **src/**: Path to all the source code
	- **data\_handler.py**: Contains functionality to load, preprocess and augment the dataset
	- **dlc\_bci.py**: Course provided loader of the BCI dataset
	- **models.py**: Contains the PyTorch LSTM and CNN model architectures
	- **plots.py**: Contains functionality to plot training and cross validation loss and accuracy curves
	- **train\_best\_cnn.py**: Script to train and test best CNN architecture
	- **train\_best\_lstm.py**: Script to train and test best LSTM architecture
	- **train\_cnn.py**: Gridsearch running cross validation over parameter configurations to find best CNN architecture
	- **train\_lstm.py**: Gridsearch running cross validation over parameter configurations to find best LSTM architecture
	- **train\_mlp.py**: Multilayer perceptron baseline
	- **train\_utils.py**: Contains training and K-fold cross validation helper functions
- **data\_bci/**: Path where data will automatically be downloaded to on the first run
- **README.md**

## How to run


``` $ cd src & python train_best_lstm.py ```