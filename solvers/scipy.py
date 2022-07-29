from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.optimize import minimize


class Solver(BaseSolver):
    """Scipy optimizers."""

    name = "scipy"

    install_cmd = 'conda'
    requirements = [
        'numpy',
        'scipy'
    ]
    parameters = {
        'solver': [
            'Nelder-Mead', 'Powell'
        ],
    }

    stopping_criterion = SufficientProgressCriterion(
        patience=5, strategy='tolerance')

    def skip(self, function, dimension):
        return False, ""
        # if dimension > 1:
        #     return True, "Bisection only runs for 1D problems"

    def set_objective(self, function, dimension):
        self.function = function
        self.dimension = dimension

    def run(self, tol):
        f = self.function
        x0 = np.ones(self.dimension) / 2.0 
        result = minimize(f, x0=x0, method=self.solver, tol=tol)
        self.xopt = result.x

    def get_result(self):
        return self.xopt
