"""Example showing how to use Luigi to build models trained on increasing subsets of data."""
import os
import time
import pickle as pkl

import luigi
import sklearn
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import json

pjoin = os.path.join
scratch_dir = "02-luigi-runs"

# The DEBUG level logging is quite verbose
luigi.interface.core.log_level="INFO"

class MasterWorkflow(luigi.WrapperTask):
    """One workflow to rule them all.
    
    This dummy task is responsible for orchestrating the entire DAG. It's a WrapperTask since
    it doesn't actually have any output itself.
    """
    def requires(self):
        begin_year = 2000
        end_year = 2012
        years = list(range(begin_year, end_year + 1))
        n_years = len(years)

        tree_config_tasks = {yr: TreeConfiguration(seed=yr) for yr in years[1:]}
        ridge_config_tasks = {yr: RidgeConfiguration(seed=yr) for yr in years[1:]}
        lasso_config_tasks = {yr: LassoConfiguration(seed=yr) for yr in years[1:]}

        data_tasks = [NumpyData(n_samples=int(10000 * (i + 1)/ n_years), n_feats=10, year=yr) for i, yr in enumerate(years)]

        ensemble_tasks = []
        for i, yr in enumerate(years[1:], start=1):
            tree_task = FitTree(config_task=tree_config_tasks[yr], data_tasks=data_tasks[:i], year=yr)
            ridge_task = FitRidge(config_task=ridge_config_tasks[yr], data_tasks=data_tasks[:i], year=yr)
            lasso_task = FitLasso(config_task=lasso_config_tasks[yr], data_tasks=data_tasks[:i], year=yr)

            ptask = EnsembleModels(model_fit_tasks=[tree_task, ridge_task, lasso_task], data_task=data_tasks[i], year=yr)
            ensemble_tasks.append(ptask)

        return ensemble_tasks


class TreeConfiguration(luigi.Task):
    seed = luigi.IntParameter()

    def output(self):
        """Create a configuration for a single job.

        :return: the target output for this task.
        """
        config_dir = pjoin(scratch_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)
        return luigi.LocalTarget(pjoin(config_dir, f'tree_config_{self.seed}.json'))

    def run(self):
        np.random.seed(self.seed)
        lr = 10 ** np.random.uniform(low=-5, high=-1)
        n_trees = int(np.random.choice([5, 10, 15, 20]))
        config = {
            "n_estimators": n_trees,
            "learning_rate": lr,
        }

        with self.output().open('w') as f:
            json.dump(config, f)

class RidgeConfiguration(luigi.Task):
    seed = luigi.IntParameter()

    def output(self):
        """Create a configuration for a single job.

        :return: the target output for this task.
        """
        config_dir = pjoin(scratch_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)
        return luigi.LocalTarget(pjoin(config_dir, f'ridge_config_{self.seed}.json'))

    def run(self):
        np.random.seed(self.seed)
        alpha = 10 ** np.random.uniform(low=-5, high=-1)
        config = {
            "alpha": alpha,
        }

        with self.output().open('w') as f:
            json.dump(config, f)

class LassoConfiguration(luigi.Task):
    seed = luigi.IntParameter()

    def output(self):
        """Create a configuration for a single job.

        :return: the target output for this task.
        """
        config_dir = pjoin(scratch_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)
        return luigi.LocalTarget(pjoin(config_dir, f'lasso_config_{self.seed}.json'))

    def run(self):
        np.random.seed(self.seed)
        alpha = 10 ** np.random.uniform(low=-5, high=-1)
        config = {
            "alpha": alpha,
        }

        with self.output().open('w') as f:
            json.dump(config, f)

class NumpyData(luigi.Task):
    n_feats = luigi.IntParameter()
    n_samples = luigi.IntParameter()
    year = luigi.IntParameter()

    def output(self):
        """Produce data files for fitting.

        :return: the predictor and response data for this task.
        """
        data_dir = pjoin(scratch_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        return {
            "X": luigi.LocalTarget(pjoin(data_dir, f'predictors_{self.year}.npy')),
            "y": luigi.LocalTarget(pjoin(data_dir, f'responses_{self.year}.npy')),
        }

    def run(self):
        X = np.random.randn(self.n_samples, self.n_feats).astype(np.float32)
        A = np.random.randint(low=-5, high=5, size=(self.n_feats, 1)).astype(np.float32)
        noise = np.random.randn(self.n_samples, 1).astype(np.float32)

        y = X @ A + noise
        np.save(self.output()["X"].path, X)
        np.save(self.output()["y"].path, y)
        

class FitRidge(luigi.Task):
    config_task = luigi.TaskParameter() 
    data_tasks = luigi.Parameter() 
    year = luigi.IntParameter() 

    def requires(self):
        return {
            "config": self.config_task,
            "data": self.data_tasks,
        }

    def output(self):
        """Produce model fit to data.

        :return: the fitted model file.
        """
        model_dir = pjoin(scratch_dir, "model_dir")
        os.makedirs(model_dir, exist_ok=True)

        return luigi.LocalTarget(pjoin(model_dir, f"ridge_model_{self.year}.pkl"))

    def run(self):
        X = np.concatenate([np.load(dtask["X"].path) for dtask in self.input()["data"]], axis=0)
        y = np.concatenate([np.load(dtask["y"].path) for dtask in self.input()["data"]], axis=0)

        with self.input()["config"].open("r") as f:
            param = json.load(f)

        model = Ridge(alpha=param["alpha"])
        with open(self.output().path, "w") as f:
            pkl.dump(model, f)

class FitLasso(luigi.Task):
    config_task = luigi.TaskParameter() 
    data_tasks = luigi.Parameter() 
    year = luigi.IntParameter() 

    def requires(self):
        return {
            "config": self.config_task,
            "data": self.data_tasks,
        }

    def output(self):
        """Produce model fit to data.

        :return: the fitted model file.
        """
        model_dir = pjoin(scratch_dir, "model_dir")
        os.makedirs(model_dir, exist_ok=True)

        return luigi.LocalTarget(pjoin(model_dir, f"lasso_model_{self.year}.pkl"))

    def run(self):
        X = np.concatenate([np.load(dtask["X"].path) for dtask in self.input()["data"]], axis=0)
        y = np.concatenate([np.load(dtask["y"].path) for dtask in self.input()["data"]], axis=0)

        with self.input()["config"].open("r") as f:
            param = json.load(f)

        model = Lasso(alpha=param["alpha"])
        with open(self.output().path, "w") as f:
            pkl.dump(model, f)


class FitTree(luigi.Task):
    config_task = luigi.TaskParameter() 
    data_tasks = luigi.Parameter() 
    year = luigi.IntParameter() 

    def requires(self):
        return {
            "config": self.config_task,
            "data": self.data_tasks,
        }

    def output(self):
        """Produce model fit to data.

        :return: the fitted model file.
        """
        model_dir = pjoin(scratch_dir, "model_dir")
        os.makedirs(model_dir, exist_ok=True)

        return luigi.LocalTarget(pjoin(model_dir, f"tree_model_{self.year}.pkl"))

    def run(self):
        X = np.concatenate([np.load(dtask["X"].path) for dtask in self.input()["data"]], axis=0)
        y = np.concatenate([np.load(dtask["y"].path) for dtask in self.input()["data"]], axis=0)

        with self.input()["config"].open("r") as f:
            param = json.load(f)

        model = GradientBoostingRegressor(**param)

        with open(self.output().path, "w") as f:
            pkl.dump(model, f)

class EnsembleModels(luigi.Task):
    model_fit_tasks = luigi.Parameter()
    data_task = luigi.TaskParameter()
    year = luigi.IntParameter()

    def requires(self):
        return {
            "models": self.model_fit_tasks,
            "data": self.data_task,
        }
    
    def output(self):
        model_dir = pjoin(scratch_dir, "forecasts")
        os.makedirs(model_dir, exist_ok=True)

        return luigi.LocalTarget(pjoin(model_dir, f"forecasts_{self.year}.npy"))

    def run(self):
        X_test = np.load(self.input()["data"]["X"].path)
        n_samples, _ = X_test.shape

        y_hat = np.zeros(n_samples)
        for model_target in self.input()["models"]:
            with open(model_target.path, "r") as f:
                model = pkl.load(f)
                y_hat += model.predict(X_test)

        np.save(self.output().path, y_hat)


"""
This is a much more maintainable way to use luigi for model building. First thing
to note is the use of a dummy task that runs the data / config generation, model
fitting, and forecasting tasks. The second thing to note is that rather than creating
each upstream dependency in `requires`, we just pass the required tasks themselves as 
parameters. This way only the dummy task needs to know about all the parameters and a
task like Predict, for example, doesn't need a ton of passthrough parameters about the
model configuration.

It does not seem luigi is meant to be used this way since passing a list of tasks gives
a warning when running the code but this is really the only sensible solution.
"""
if __name__ == '__main__':
    luigi.run()
    # from command line run the following
    # PYTHONPATH='.' luigi --module 01-simple MasterWorkflow