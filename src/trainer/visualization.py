import logging
import os

import matplotlib
import matplotlib.pyplot as plt

from utils.file_utils import process_path

matplotlib.use('agg')  # because of problems with tkinter

log = logging.getLogger("visualization")
logging.getLogger('matplotlib.font_manager').disabled = True


def plot_graphs(history, metric: str) -> None:
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])


def build_loss_acc(history, file_path: str) -> None:
    log.info("Building loss and accuracy graphs...")
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)
    plt.savefig(os.getcwd() + file_path + '\\loss_acc')
    log.info("Graphs saved!")


def build_f1(history, path: str) -> None:
    log.info("Building f1 graph...")
    plt.figure()
    plt.title('F1 (by epochs)')
    plt.plot(history.history['f1'], label='Train')
    plt.plot(history.history['val_f1'], label='Test')
    plt.legend()
    plt.savefig(path)


def build_graphs(history, file_path: str) -> None:
    log.info("Building loss, accuracy, precision, recall graphs...")
    plt.figure(figsize=(16, 16))
    plt.subplot(2, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(2, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)
    plt.subplot(2, 2, 3)
    plot_graphs(history, 'precision')
    plt.ylim(None, 1)
    plt.subplot(2, 2, 4)
    plot_graphs(history, 'recall')
    plt.ylim(None, 1)
    file_path = process_path(file_path, make_dirs=True)
    plt.savefig(os.path.join(file_path, 'graphs.png'))
    build_f1(history, os.path.join(file_path, 'f1.png'))
    log.info("Graphs saved!")
