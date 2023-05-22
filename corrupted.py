import torch
import torchvision
import torchvision.datasets as datasets

import numpy as np

class CIFAR10RandomLabels(datasets.CIFAR10):
    '''
    Use with pre-saved corrupted datasets
    '''
    def __init__(self, corrupt_prob=0.0, num_classes=10,**kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]
        self.targets = labels

class CorruptedCIFAR10(torchvision.datasets.CIFAR10):
    '''
    Generate corrupted dataset with a fixed seed for reproducibility
    '''
    def __init__(self, prob = 0.0, seed = 0, return_index = False, **argv):
        super().__init__(**argv)
        
        self.return_index = return_index

        if prob > 0.0:
            rng = np.random.default_rng(seed)
            
            corrupt = rng.random(size = len(self.targets)) < prob
            for i, (l, c) in enumerate(zip(self.targets, corrupt)):
                if c:
                    new_label = rng.integers(9, dtype=int)
                    self.targets[i] = new_label if new_label < l else new_label+1
        
    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        
        if self.return_index:
            return data, target, index
        else:
            return data, target
