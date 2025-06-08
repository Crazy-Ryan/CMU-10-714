import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.transforms = transforms
        images = []
        labels = []
        data_batch_files = [f'data_batch_{i}' for i in range(1, 6)] if train else ['test_batch']
        for filename in data_batch_files:
            full_path = os.path.join(base_folder, filename)
            with open(full_path, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                images.append(dict[b'data']/255)
                labels.append(np.array(dict[b'labels']))
        self.X = np.concatenate(images, axis=0)
        self.y = np.concatenate(labels, axis=0)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return self.X[index].reshape(3,32,32), self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return self.y.shape[0]
        ### END YOUR SOLUTION
