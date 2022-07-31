from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.optimize import basinhopping


class Solver(BaseSolver):
    """Scipy basinhopping."""

    name = "basinhopping"

    install_cmd = "conda"
    requirements = ["numpy", "scipy"]
    parameters = {
        "temperature": [1, 10],
        "seed": [42],
    }

    def set_objective(self, function, dimension, bounds):
        self.function = function
        self.dimension = dimension
        self.bounds = bounds

    def run(self, n_iter):
        f = self.function
        rng = np.random.RandomState(self.seed)  # fix seed
        x0 = rng.uniform(size=self.dimension,
                         low=self.bounds[0],
                         high=self.bounds[1])
        if n_iter == 0:
            self.xopt = x0
            return
        result = basinhopping(f, x0=x0, niter=n_iter - 1, T=self.temperature)
        self.xopt = result.x

    def get_result(self):
        return self.xopt.flatten()
