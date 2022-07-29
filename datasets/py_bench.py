from benchopt import safe_import_context
from benchopt import BaseDataset

with safe_import_context() as import_ctx:
    import numpy as np
    from PyBenchFCN import SingleObjectiveProblem as SOP


class Dataset(BaseDataset):

    name = "FCN"

    install_cmd = 'conda'
    requirements = ['pip:PyBenchFCN']

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        # "function": ["ackley", "rosenbrock", "rastrigin", "schwefel"],
        "function": ["ackley", "rastrigin", "rosenbrock"],
        "dimension": [10],
        # "dimension": [2, 3],
    }

    def __init__(self, function="ackley", dimension=2):
        self.function = function
        self.dimension = dimension

    def get_data(self):
        if self.function == "ackley":
            problem = SOP.ackleyfcn(self.dimension)      # Ackley problem
        elif self.function == "rosenbrock":
            problem = SOP.rosenbrockfcn(self.dimension)   # Rosenbrock problem
        elif self.function == "rastrigin":
            problem = SOP.rastriginfcn(self.dimension)   # Rastrigin problem
        elif self.function == "schwefel":
            problem = SOP.schwefel220fcn(self.dimension)    # Schwefel problem
        else:
            raise NotImplementedError(
                f"Function {self.function} not implemented"
            )
        return dict(function=problem.f, dimension=problem.n_var)
