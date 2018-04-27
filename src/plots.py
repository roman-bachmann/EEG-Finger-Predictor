import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 28
plt.style.use('ggplot')
plt.rcParams["axes.grid"] = False
c = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['figure.figsize'] = 8, 4


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    """
    Plots both the training and validation losses and accuracies.

    Args:

        train_losses (list[float]): List of training losses

        val_losses (list[float]): List of validation losses

        train_accs (list[float]): List of training accuracies

        val_accs (list[float]): List of validation accuracies
    """
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, '--', c=c[0], label='Train loss')
    plt.plot(val_losses, c=c[0], label='Val loss')
    plt.legend(loc='best')

    plt.figure(figsize=(8,4))
    plt.plot(train_accs, '--', c=c[1], label='Train acc')
    plt.plot(val_accs, c=c[1], label='Val acc')
    plt.legend(loc='best')

    plt.show()


def plot_CV_learning_curves(K_train_losses, K_val_losses, K_train_accs, K_val_accs):
    """
    Plots all the training and validation losses and accuracies for a K-fold
    Cross Validation. Highlights the mean of all K folds.

    Args:

        K_train_losses (list[list[float]]): List of all K training losses lists

        K_val_losses (list[list[float]]): List of all K validation losses lists

        K_train_accs (list[list[float]]): List of all K training accuracies lists

        K_val_accs (list[list[float]]): List of all K validation accuracies lists
    """
    plt.figure(figsize=(8,4))
    for train_losses in K_train_losses:
        plt.plot(train_losses, '--', c=c[0], label='Train loss CV', alpha=0.4)
    for val_losses in K_val_losses:
        plt.plot(val_losses, c=c[0], label='Val loss CV', alpha=0.4)
    mean_train_losses = np.array(K_train_losses).mean(axis=0)
    mean_val_losses = np.array(K_val_losses).mean(axis=0)
    plt.plot(mean_train_losses, '--', c=c[0], label='Train loss mean')
    plt.plot(mean_val_losses, c=c[0], label='Val loss mean')
    plt.legend(loc='best')

    plt.figure(figsize=(8,4))
    for train_accs in K_train_accs:
        plt.plot(train_accs, '--', c=c[1], label='Train acc CV', alpha=0.4)
    for val_accs in K_val_accs:
        plt.plot(val_accs, c=c[1], label='Val acc CV', alpha=0.4)
    mean_train_accs = np.array(K_train_accs).mean(axis=0)
    mean_val_accs = np.array(K_val_accs).mean(axis=0)
    plt.plot(mean_train_accs, '--', c=c[1], label='Train acc mean')
    plt.plot(mean_val_accs, c=c[1], label='Val acc mean')
    plt.legend(loc='best')

    plt.show()
