import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager

#Defaults for legible figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams["image.cmap"] = 'jet'

timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))

def plot_metrics(history, acc='accuracy', loss='loss', 
                 val_acc='val_accuracy', val_loss='val_loss', show=False,
                 save_png=True, outpath='training_curves_' + timestamp()):
    all_keys = [acc, loss, val_acc, val_loss]
    keys = list(history.history)
    idx = np.asarray([k in keys for k in all_keys])
    np.asarray(all_keys)[idx]
    keys = list(history.history)
    epochs = range(len(history.history[keys[0]]))
    plt.figure(figsize=(8,8))
    if acc in keys:
        acc  = history.history[acc]
        plt.subplot(211)
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
    if val_acc in keys:
        val_acc = history.history[val_acc]
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    if loss in keys:
        loss = history.history[loss]
        plt.subplot(212)
        plt.plot(epochs, loss, 'bo', label='Training Loss')
    if val_loss in keys:
        val_loss = history.history[val_loss]
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    if save_png:
        plt.savefig(outpath)
    if show:
        plt.show()