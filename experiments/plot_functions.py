import matplotlib.pyplot as plt
import tensorflow as tf
import os
def plot_logs_classification(EXPERIMENT_NAME: str, history_1, history_2=None, show=False):
    """
    Assumes keras history objects to have logs accuracy,
    val_accuracy, ce, val_ce, lbe, val_lbe, max_lbe, val_max_lbe, loss, and val_loss.
    """
    os.makedirs(f"experiments/results/{EXPERIMENT_NAME}/", exist_ok=True)

    with_lbe_history = history_1
    if history_2:
        without_lbe_history = history_2

    plt.rcParams['savefig.facecolor']='white'
    # plot accuracies
    if history_2:
        plt.plot(without_lbe_history.history['accuracy'])
        plt.plot(without_lbe_history.history['val_accuracy'])
    plt.plot(with_lbe_history.history['accuracy'])
    plt.plot(with_lbe_history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    if history_2:
        labels = ['noLBE_train', 'noLBE_validation',"LBE_train","LBE_val"]
    else:
        labels = ["LBE_train","LBE_val"]
    plt.legend(labels, loc='upper left')
    plt.savefig(f"experiments/results/{EXPERIMENT_NAME}/accuracies.svg")

    plt.close()

    # plot categorical crossentropy loss
    if history_2:
        plt.plot(without_lbe_history.history['loss'])
        plt.plot(without_lbe_history.history['val_loss'])
    plt.plot(with_lbe_history.history['ce'])
    plt.plot(with_lbe_history.history['val_ce'])
    plt.title('Loss (categorical cross-entropy)')
    plt.ylabel('CE loss')
    plt.xlabel('epoch')
    plt.legend(labels, loc='upper left')
    plt.savefig(f"experiments/results/{EXPERIMENT_NAME}/crossentropy.svg")

    plt.close()

    # plot average layer batch entropy error for each epoch
    plt.plot(with_lbe_history.history['lbe'])
    plt.plot(with_lbe_history.history['val_lbe'])
    plt.title('Average Regularization Loss (LBE)')
    plt.ylabel('Avg LBE loss')
    plt.xlabel('Epoch')
    plt.legend(["LBE_train","LBE_val"], loc='upper left')
    plt.savefig(f"experiments/results/{EXPERIMENT_NAME}/LBE_loss.svg")

    plt.close()

    # plot maximal layer batch entropy error for each epoch
    plt.plot(with_lbe_history.history["max_lbe"])
    plt.plot(with_lbe_history.history['val_max_lbe'])
    plt.title('Maximal Regularization Loss (LBE)')
    plt.ylabel('Max LBE loss')
    plt.xlabel('Epoch')
    plt.legend(["LBE_train","LBE_val"], loc='upper left')
    plt.savefig(f"experiments/results/{EXPERIMENT_NAME}/Max_LBE_loss.svg")

    plt.close()
