from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    import optuna
    from optuna import samplers


class Solver(BaseSolver):
    """optuna"""

    name = "optuna"

    install_cmd = "conda"
    requirements = [
        "optuna",
    ]
    parameters = {
        "solver": ["TPE", "RandomSearch"],
    }

    stopping_criterion = SufficientProgressCriterion(
        patience=7, strategy='iteration')

    def skip(self, function, dimension):
        return False, ""

    def set_objective(self, function, dimension):
        self.function = function
        self.dimension = dimension

    def run(self, n_iter):
        if n_iter == 0:
            self.xopt = np.ones(self.dimension) / 2.0
            return

        def objective(trial):
            x = np.array([
                trial.suggest_uniform(f'x_{k}', -1, 2)
                for k in range(self.dimension)
            ])
            return self.function(x)

        if self.solver == "TPE":
            sampler = samplers.TPESampler(seed=10, n_startup_trials=10)
        elif self.solver == "RandomSearch":
            sampler = samplers.RandomSampler(seed=10)
        else:
            raise NotImplementedError(f"Solver {self.solver} not implemented")
        study = optuna.create_study(sampler=sampler, direction='minimize')
        optuna.logging.disable_default_handler()  # limit verbosity
        study.optimize(objective, n_trials=n_iter)
        self.xopt = study.best_trial.params

    def get_result(self):
        xopt = self.xopt
        if isinstance(xopt, dict):
            xopt = np.array([xopt[f'x_{k}'] for k in range(self.dimension)])
        return xopt
