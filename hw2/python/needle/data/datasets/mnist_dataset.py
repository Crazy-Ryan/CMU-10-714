import gzip
import struct
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        super().__init__(transforms)
        with (gzip.open(image_filename, 'rb') as f):
            _, _, row_pix, col_pix = struct.unpack('>iiii', f.read(16))
            raw_X = np.fromiter((num[0] for num in struct.iter_unpack("B", f.read())), dtype=np.uint8
                                ).reshape((-1, row_pix * col_pix))
            self.X = np.divide(raw_X, 255, dtype=np.float32)
        with (gzip.open(label_filename, 'rb') as f):
            f.read(8)
            self.y = np.fromiter((num[0] for num in struct.iter_unpack("B", f.read())), dtype=np.uint8)

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        res = self.X[index]
        if self.transforms is not None:
            res = self.apply_transforms(res.reshape((28, 28, 1))).flatten()
        return res, self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return self.X.shape[0]
        ### END YOUR SOLUTION