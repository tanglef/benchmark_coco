from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import nevergrad as ng


class Solver(BaseSolver):
    """nevergrad"""

    name = "nevergrad"

    install_cmd = "conda"
    requirements = [
        "nevergrad",
    ]
    parameters = {
        "solver": ["NGOpt", "RandomSearch", "ScrHammersleySearch",
                   "TwoPointsDE", "CMA", "PSO"],
    }

    def set_objective(self, function, dimension):
        self.function = function
        self.dimension = dimension

    def run(self, n_iter):
        if n_iter == 0:
            self.xopt = np.ones(self.dimension) / 2.0
            return

        f = self.function
        parametrization = ng.p.Array(shape=(self.dimension,))
        parametrization.random_state = np.random.RandomState(42)  # fix seed
        optimizer = ng.optimizers.registry[self.solver](
            budget=n_iter, parametrization=parametrization, num_workers=1
        )
        recommendation = optimizer.minimize(f)
        self.xopt = np.array(recommendation.value)

    def get_result(self):
        return self.xopt.flatten()
