"""Example showing how NOT to use Luigi."""
import os
import time

import luigi
import xgboost as xgb
import numpy as np
import json

pjoin = os.path.join
scratch_dir = "00-luigi-runs"


class Configuration(luigi.Task):
    seed = luigi.IntParameter()

    def output(self):
        """Create a configuration for a single job.

        :return: the target output for this task.
        """
        config_dir = pjoin(scratch_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)
        return luigi.LocalTarget(pjoin(config_dir, f'config_{self.seed}.json'))

    def run(self):
        np.random.seed(self.seed)
        lr = 10 ** np.random.uniform(low=-5, high=-1)
        n_trees = int(np.random.choice([5, 10, 15, 20]))
        config = {
            "num_rounds": n_trees,
            "eta": lr,
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
        

class FitTree(luigi.Task):
    seed = luigi.IntParameter() 
    n_feats = luigi.IntParameter() 
    n_samples = luigi.IntParameter() 
    year = luigi.IntParameter() 

    def requires(self):
        return {
            "config": Configuration(seed=self.seed),
            "data": NumpyData(n_feats=self.n_feats, n_samples=self.n_samples, year=self.year)
        }

    def output(self):
        """Produce model fit to data.

        :return: the fitted model file.
        """
        model_dir = pjoin(scratch_dir, "model_dir")
        os.makedirs(model_dir, exist_ok=True)

        return luigi.LocalTarget(pjoin(model_dir, f"tree_model_{self.year}.model"))

    def run(self):
        X = np.load(self.input()["data"]["X"].path)
        y = np.load(self.input()["data"]["y"].path)

        dtrain = xgb.DMatrix(X, label=y.squeeze())
        with self.input()["config"].open("r") as f:
            param = json.load(f)

        bst = xgb.train(param, dtrain)
        bst.save_model(self.output().path)


class PredictTree(luigi.Task):
    seed = luigi.IntParameter() 
    n_feats = luigi.IntParameter() 
    n_samples = luigi.IntParameter() 
    year = luigi.IntParameter() 

    def requires(self):
        return {
            "data": NumpyData(n_feats=self.n_feats, n_samples=self.n_samples, year=self.year),
            "model": FitTree(seed=self.seed, n_feats=self.n_feats, n_samples=self.n_samples, year=self.year),
        }
    
    def output(self):
        """Produce predictions on the data.

        :return: file containing predictions
        """
        model_dir = pjoin(scratch_dir, "forecasts")
        os.makedirs(model_dir, exist_ok=True)

        return luigi.LocalTarget(pjoin(model_dir, f"forecasts_{self.year}.npy"))

    def run(self):
        X_test = np.load(self.input()["data"]["X"].path)
        dtest = xgb.DMatrix(X_test)

        bst = xgb.Booster({'nthread': 1})  # init model
        bst.load_model(self.input()["model"].path)

        y_hat = bst.predict(dtest)
        np.save(self.output().path, y_hat)


"""
The problem with this way of doing things in luigi is it makes things a 
horrible mess if your tasks are more than two or three levels deep. Note
how in the PredictTree method we are passing in things like `n_feats`. 
Ideally Predict doesn't need to know about these hyperparams since only
Fit really cares, but since we're constructing the Fit task inside Predict's
`requires` method we have to pass the params in.

This method of doing things in Luigi will lead to either the last task having tons
of parameters or you needing to create a God object with every parameter that every
task would ever want. This is annoying because the last task doesn't actually need
this God object, but because the upstream tasks on which it depends need to be created 
and _they_ require some or all of these params, you have no choice.

The end result of this usage is an unmaintainable brittle mess. For some reason, this
is the only usage pattern shown in the docs...
"""
if __name__ == '__main__':
    luigi.run()
    # from command line run the following
    # PYTHONPATH='.' luigi --module 00-antipattern PredictTree --seed 314 --n-feats 10 --n-samples 1000 --year 2000
