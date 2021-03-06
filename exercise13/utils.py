import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
from torch.utils.data import Dataset


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.tensor(weights, dtype=torch.float)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            return dataset[idx][1]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word: str):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class RawDataset(Dataset):
    """
    Intentionally to use only
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.copy()
        self.len = len(self.df)

    def __getitem__(self, index):
        record = self.df.iloc[index]
        return record.Phrase, record.Sentiment

    def __len__(self):
        return self.len


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def make_var(names):
    ans = []
    for name in sorted(names, reverse=True, key=len):
        name = name.lower()
        tmp = [ord(chr) for chr in name]
        tmp = torch.tensor(tmp, dtype=torch.long)
        ans.append(tmp)
    return torch.nn.utils.rnn.pad_sequence(ans, batch_first=True)


def count_non_zero_length(aaa):
    """
    aaa = [[116, 105, 114,  97, 115],
         [ 97, 110, 110,   0,   0],
         [101, 108,   0,   0,   0]]
    ans -> [5, 3, 2]
    """
    bbb = []
    for item in aaa:
        counting = 0
        for element in item:
            if element != 0:
                counting += 1
        bbb.append(counting)
    return bbb


def str2ascii_arr(name):
    """
    0-255
    """
    arr = [ord(c) for c in name]
    return arr, len(arr)
