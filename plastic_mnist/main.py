
import numpy as np
from dataloader import MnistDataset

class ShuffledMnistDataset(MnistDataset):
    def shuffle(self):
        self.train_labels = np.random.shuffle(self.train_labels)
