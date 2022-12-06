from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    min_benchopt_version = "1.3"
    name = "Zero-order test functions"

    def get_one_solution(self):
        # Return one solution. This should be compatible with 'self.compute'.
        return np.zeros(self.dimension)

    def set_data(self, function, dimension, bounds):
        self.function = function
        self.dimension = dimension
        self.bounds = bounds

    def compute(self, x):
        return self.function(x)

    def get_objective(self):
        return dict(function=self.function,
                    dimension=self.dimension,
                    bounds=self.bounds)
