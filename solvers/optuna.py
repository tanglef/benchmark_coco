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
        "cmaes",
        "numpy",
    ]
    parameters = {
        "solver": ["cmaes", "TPE", "RandomSearch"],
    }

    stopping_criterion = SufficientProgressCriterion(
        patience=3, strategy='iteration')

    def set_objective(self, function, dimension):
        self.function = function
        self.dimension = dimension

    def run(self, n_iter):
        n_iter += 1  # no possible to call optuna with 0 trial

        def objective(trial):
            x = np.array([
                trial.suggest_float(f'x_{k}', -2, 2)
                for k in range(self.dimension)
            ])
            return self.function(x)

        seed = 10  # to make results reproducible
        if self.solver == "TPE":
            sampler = samplers.TPESampler(seed=seed, n_startup_trials=10)
        elif self.solver == "RandomSearch":
            sampler = samplers.RandomSampler(seed=seed)
        elif self.solver == "cmaes":
            sampler = samplers.CmaEsSampler(seed=seed)
        else:
            raise NotImplementedError(f"Solver {self.solver} not implemented")
        study = optuna.create_study(sampler=sampler, direction='minimize')
        optuna.logging.disable_default_handler()  # limit verbosity
        study.optimize(objective, n_trials=n_iter)
        self.xopt = study.best_trial.params

    def get_result(self):
        xopt = np.array([self.xopt[f'x_{k}'] for k in range(self.dimension)])
        return xopt
