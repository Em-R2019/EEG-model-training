import os
from glob import glob
from math import floor
from os.path import join

import numpy as np
# from scipy import signal
from skimage.measure import block_reduce
from torch.utils.data import Dataset, DataLoader
import mne as mne
from sklearn.model_selection import train_test_split


class EEGDataset(Dataset):
    """PyTorch Dataset class for our EEG datasets.
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch = self.data[idx]
        labels = self.labels[idx]
        return batch, labels

def get_event_data_labels(data, event_idx, label_dict):
    event_data = []

    for i in range(len(event_idx)):
        if event_idx[i][2] in label_dict:
            idx1 = event_idx[i][0]
            if i == len(event_idx) - 1:
                idx2 = data.shape[1]
            else:
                idx2 = event_idx[i + 1][0]

            event_data.append([data[:, idx1:idx2], label_dict[event_idx[i][2]]])
    return event_data

def get_label_dict(event_dict, pos, neg):
    label_dict = {}

    if isinstance(pos, str):
        pos = [pos]
    if isinstance(neg, str):
        neg = [neg]

    for pos_label in pos:
        if pos_label in event_dict:
            label_dict[event_dict[pos_label]] = 1
    for neg_label in neg:
        if neg_label in event_dict:
            label_dict[event_dict[neg_label]] = 0
    return label_dict

def segment_data(data, dt):
    segmented_data = []
    labels = []

    channels = 64

    dsample = round(1/dt)
    overlap = round(0.5/dt)
    dsample_start = dsample - overlap

    for task in data:
        task_label = task[1]
        task_data = task[0][0:channels, :]

        nsegments = len(task_data[1]) / dsample
        nsegments = floor(nsegments * (dsample / overlap) - 1)

        for i in range(nsegments):
            segmented_data.append(task_data[:,i * dsample_start:i * dsample_start + dsample])
            labels.append(task_label)

    return segmented_data, labels


def process_data(file_path, pos, neg):
    data = []
    labels = []

    # sos_low = None

    data_path = os.path.join(file_path, "*.fif")
    for file in glob(data_path, recursive=True):
        raw = mne.io.read_raw_fif(file, verbose=False, preload=True)

        # if sos_low is None:
        #     sos_low = signal.butter(10, 40, 'lowpass', fs=250, output='sos')

        file_data = raw.get_data(picks=range(1, 19))

        # file_data = signal.sosfiltfilt(sos_low, file_data, axis=1)
        file_data = block_reduce(file_data, block_size=(1,2), func=np.mean, cval=np.mean(file_data))

        event_idx, event_dict = mne.events_from_annotations(raw, verbose=False)

        event_idx[:, 0] = event_idx[:, 0] // 2 # Downsample to 250 Hz

        label_dict = get_label_dict(event_dict, pos, neg)

        file_data = get_event_data_labels(file_data, event_idx, label_dict)

        if len(file_data) > 0:
            segmented_data, segment_labels = segment_data(file_data, raw.times[1]*2)
            data.extend(segmented_data)
            labels.extend(segment_labels)

    return data, labels

def augment_data(data, scale):
    aug_data = []
    for segment in data:
        noise = np.random.normal(0,scale, segment.shape)
        aug_data.append(segment + noise)
    return aug_data


def load(file_path, batch_size, pos, neg, shuffle=True, augment=0):
    data, labels = process_data(file_path, pos, neg)

    full_train_data, test_data, train_val_labels, test_labels = train_test_split(
        data, labels,
        stratify=labels,
        test_size=0.2
    )

    train_data, val_data, train_labels, val_labels = train_test_split(
        full_train_data, train_val_labels,
        stratify=train_val_labels,
        test_size=0.2
    )

    if augment > 0.:
        aug_data = augment_data(train_data, augment)
        train_data.extend(aug_data)
        train_labels.extend(train_labels)

        aug_data = augment_data(full_train_data, augment)
        full_train_data.extend(aug_data)
        train_val_labels.extend(train_val_labels)

    train_set = EEGDataset(train_data, train_labels)
    val_set = EEGDataset(val_data, val_labels)
    test_set = EEGDataset(test_data, test_labels)

    train_val_set = EEGDataset(full_train_data, train_val_labels)

    trainloader = DataLoader(train_set, batch_size, shuffle=shuffle, num_workers=0)
    valloader = DataLoader(val_set, batch_size, shuffle=shuffle, num_workers=0)
    testloader = DataLoader(test_set, batch_size, shuffle=shuffle, num_workers=0)

    train_val_loader = DataLoader(train_val_set, batch_size, shuffle=shuffle, num_workers=0)

    return trainloader, valloader, testloader, train_val_loader

if __name__ == "__main__":
    path = join("data", "S3", '*')
    load(path, 24, pos=['MI', 'MM'], neg='Rest')