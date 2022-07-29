import numpy as np


from benchopt import BaseSolver


class Solver(BaseSolver):
    """Bisector method."""

    name = "Bisector"

    # any parameter defined here is accessible as a class attribute
    parameters = {"min0": [-10.], "max0": [10.]}

    def skip(self, function, dimension):
        if dimension > 1:
            return True, "Bisection only runs for 1D problems"

    def set_objective(self, function, dimension):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.function = function
        self.dimension = dimension

    def run(self, n_iter):
        f = self.function
        a, b = self.min0, self.max0
        xdown = min(a, b)
        xup = max(a, b)
        for(i in range(n_iter)):
            xbar = 1 / 2 * (xup + xdown)
            fxbar = f(xbar)
            if f(xdown) * f(xbar) <= 0:
                xup = xbar
            else:
                xdown = xbar
        self.xopt = xbar

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.xopt
