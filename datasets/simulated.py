from benchopt import safe_import_context
from benchopt import BaseDataset

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "simulated"
    install_cmd = "conda"
    requirements = ["numpy"]

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        "dimension": [2, 10],
    }

    def __init__(self, dimension=2):
        self.function = lambda x: np.linalg.norm(x, 2) ** 2
        self.dimension = dimension

    def get_data(self):
        return dict(function=self.function, dimension=self.dimension)
