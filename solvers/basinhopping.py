from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.optimize import basinhopping


class Solver(BaseSolver):
    """Scipy basinhopping."""

    name = "basinhopping"

    install_cmd = 'conda'
    requirements = [
        'numpy',
        'scipy'
    ]
    parameters = {
        'temperature': [
            1, 10
            # 1e-2, 1e-1, 1, 10
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
        x0 = np.ones(self.dimension) / 2.0
        if n_iter == 0:
            self.xopt = x0
            return
        result = basinhopping(f, x0=x0, niter=n_iter-1, T=self.temperature)
        self.xopt = result.x

    def get_result(self):
        return self.xopt
