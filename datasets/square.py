from benchopt import safe_import_context
from benchopt import BaseDataset

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "square"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {"dimension": [1, 2, 3]}

    def __init__(self, dimension):
        self.dimension = dimension

    def get_data(self):
        def f(x):
            return np.linalg.norm(x, 2)

        return dict(function=f, dimension=self.dimension)
