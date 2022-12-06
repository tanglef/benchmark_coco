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
        "seed": [42],
    }

    def set_objective(self, function, dimension, bounds):
        self.function = function
        self.dimension = dimension
        self.bounds = bounds

    def run(self, n_iter):
        rng = np.random.RandomState(self.seed)  # fix seed

        if n_iter == 0:
            x0 = rng.uniform(size=self.dimension,
                             low=self.bounds[0],
                             high=self.bounds[1])
            self.xopt = x0
            return

        f = self.function
        parametrization = ng.p.Array(shape=(self.dimension,))
        parametrization.set_bounds(self.bounds[0], self.bounds[1])
        parametrization.random_state = rng  # fix seed
        optimizer = ng.optimizers.registry[self.solver](
            budget=n_iter, parametrization=parametrization, num_workers=1
        )
        recommendation = optimizer.minimize(f)
        self.xopt = np.array(recommendation.value)

    def get_result(self):
        return self.xopt.flatten()
