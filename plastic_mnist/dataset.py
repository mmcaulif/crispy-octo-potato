import gzip
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn.functional as F


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # Skip magic number and read dimensions
        f.read(4)
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')

        # Read the image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # Skip magic number and read number of labels
        f.read(4)
        num_labels = int.from_bytes(f.read(4), 'big')

        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


class MnistDataset(Dataset):
    def __init__(self, train, transform=None, target_transform=None):       

        # Replace 'path/to/train-images-idx3-ubyte.gz' with the path to your MNIST training image file
        self.train_images = load_mnist_images('plastic_mnist/mnist/train-images-idx3-ubyte.gz').copy()
        # Replace 'path/to/train-labels-idx1-ubyte.gz' with the path to your MNIST training label file
        self.train_labels = load_mnist_labels('plastic_mnist/mnist/train-labels-idx1-ubyte.gz').copy()

        np.random.shuffle(self.train_labels)
        # print(self.train_labels)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        image = np.array(self.train_images[idx])
        # print(image.shape)
        label = np.array(self.train_labels[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        # print(label)
        label = np.eye(10)[label]
        # print(label)
        image = image.flatten()
        return image, label


def main():
    train_data = MnistDataset(True)
    train_dataloader = DataLoader(train_data, batch_size=1024, shuffle=True)

    for data in train_dataloader:
        train_images, train_labels = data

        # print(train_images.shape, train_labels.shape)

    # # Display an example image
    # plt.imshow(train_images[0][0], cmap='gray')
    # plt.title(f"Label: {train_labels[0]}")
    # plt.show()

if __name__ == '__main__':
    main()