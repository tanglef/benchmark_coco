from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import nevergrad as ng


class Solver(BaseSolver):
    """nevergrad"""

    name = "nevergrad"

    install_cmd = 'conda'
    requirements = [
        'nevergrad',
    ]
    parameters = {
        'solver': [
            'CMA'
        ],
    }

    def skip(self, function, dimension):
        return False, ""
        # if dimension > 1:
        #     return True, "Bisection only runs for 1D problems"

    def set_objective(self, function, dimension):
        self.function = function
        self.dimension = dimension

    def run(self, n_iter):
        f = self.function
        parametrization = ng.p.Array(shape=(self.dimension,))
        if self.solver == "CMA":
            optimizer = ng.optimizers.CMA(budget=n_iter,
                                          parametrization=parametrization,
                                          num_workers=1)
        else:
            raise NotImplementedError("Solver not implemented")
        recommendation = optimizer.minimize(f)
        # import ipdb; ipdb.set_trace()
        self.xopt = recommendation.value

    def get_result(self):
        return self.xopt
