"""Training callbacks and plotting helpers shared by the NTU training scripts."""

import itertools

import keras
import numpy as np
from keras import backend as K


def _plt():
    """Import pyplot on first use, with the non-interactive backend selected.

    Kept lazy so that importing `LrReducer` -- a plain Keras callback with no
    plotting involved -- does not drag in matplotlib and its image stack.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    return plt


class LrReducer(keras.callbacks.Callback):
    '''Learning-rate schedule used in the second training stage.

    Matches the strategy described in Sec. IV-B of the paper: when the
    validation loss stops improving for `patience` epochs the learning rate is
    divided by 10, up to `reduce_nb` times, after which training stops early.
    '''

    def __init__(self, patience=5, reduce_rate=0.1, reduce_nb=10, verbose=1):
        super(LrReducer, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_loss = 100.0
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs.get('val_loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
            if self.verbose > 0:
                print('---current best val loss: %.3f' % current_loss)
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.current_reduce_nb <= self.reduce_nb:
                    lr = K.get_value(self.model.optimizer.lr)
                    K.set_value(self.model.optimizer.lr, lr * self.reduce_rate)
                    print("reduce lr by dividing 10x")
                else:
                    if self.verbose > 0:
                        print("Epoch %d: early stopping" % (epoch))
                    self.model.stop_training = True
            else:
                if self.verbose > 0:
                    print("current loss > best_loss, but doesn't reach the patience epochs")
            self.wait += 1


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt = _plt()
    if cmap is None:
        cmap = plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def visualize_layer(model, layer_name, data, time_step):
    """Render every channel of `layer_name`'s feature map at one time step.

    Used to produce the SSI / filter-response figures in the paper.
    """
    plt = _plt()
    import colormaps as cmaps

    # NB: this used to read a module-level `model_scale3` instead of the
    # `model` argument, so it only ever worked for that one global.
    layer_map = K.function([model.layers[0].input], [model.get_layer(layer_name).output])
    layer_output = layer_map([data])[0]
    f1 = layer_output[0, time_step, :, :, :]
    width = int(np.sqrt(f1.shape[-1]))
    for channel in range(f1.shape[-1]):
        img_show = f1[:, :, channel]
        plt.subplot(width, width, channel + 1)
        plt.imshow(img_show, cmap=cmaps.parula)
        plt.axis('off')
    plt.show()


def plot_training_history(history, filename):
    """Plot accuracy / loss curves of a completed `fit()` call.

    `history` may also be a list of History objects from consecutive training
    stages (e.g. the Adam stage followed by the SGD stage); their curves are
    concatenated end to end.

    Validation curves are included when every stage used `validation_split`.
    """
    plt = _plt()
    histories = history if isinstance(history, (list, tuple)) else [history]

    def series(key):
        out = []
        for h in histories:
            out += h.history[key]
        return out

    has_val = all('val_acc' in h.history for h in histories)

    class _Merged(object):
        history = {k: series(k) for k in ('acc', 'loss')}

    merged = _Merged()
    if has_val:
        merged.history['val_acc'] = series('val_acc')
        merged.history['val_loss'] = series('val_loss')
    history = merged

    plt.style.use('ggplot')

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.plot(history.history['acc'], label='Train')
    if has_val:
        ax1.plot(history.history['val_acc'], label='Validation')
    ax1.legend(loc='upper left')

    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')
    ax2.plot(history.history['loss'], label='Train')
    if has_val:
        ax2.plot(history.history['val_loss'], label='Validation')
    ax2.legend(loc='upper left')

    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax3.set_title('Loss vs Accuracy')
    ax3.set_xlabel('Loss')
    ax3.set_ylabel('Accuracy')
    ax3.plot(history.history['loss'], history.history['acc'])

    plt.tight_layout()
    plt.savefig(filename)
