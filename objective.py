from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "COCO test functions"

    # All parameters 'p' defined here are available as 'self.p'

    def get_one_solution(self):
        # Return one solution. This should be compatible with 'self.compute'.
        return np.zeros(self.dimension)

    def set_data(self, function, dimension):
        self.function = function
        self.dimension = dimension

    def compute(self, x):
        return self.function(x)

    def to_dict(self):
        return dict(function=self.function, dimension=self.dimension)
