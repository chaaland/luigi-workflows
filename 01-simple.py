"""Example showing how to use Luigi to build models trained on subsets of data."""
import os
import time

import luigi
import xgboost as xgb
import numpy as np
import json

pjoin = os.path.join
scratch_dir = "01-luigi-runs"


class MasterWorkflow(luigi.WrapperTask):
    """One workflow to rule them all.
    
    This dummy task is responsible for orchestrating the entire DAG. It's a WrapperTask since
    it doesn't actually have any output itself.
    """
    def requires(self):
        begin_year = 2000
        end_year = 2010
        years = list(range(begin_year, end_year + 1))
        n_years = len(years)

        config_tasks = {yr: Configuration(seed=yr) for yr in years[1:]}
        data_tasks = [NumpyData(n_samples=int(10000 * (i + 1)/ n_years), n_feats=10, year=yr) for i, yr in enumerate(years)]

        prediction_tasks = []
        for i, yr in enumerate(years[1:], start=1):
            ftask = FitTree(config_task=config_tasks[yr], data_tasks=data_tasks[:i], year=yr)
            ptask = PredictTree(fit_task=ftask, data_task=data_tasks[i], year=yr)
            prediction_tasks.append(ptask)

        return prediction_tasks


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

        return luigi.LocalTarget(pjoin(model_dir, f"tree_model_{self.year}.model"))

    def run(self):
        sleep_time = np.random.choice([3, 6, 9, 12])
        self.set_status_message(f"Loading predictors!")
        self.set_progress_percentage(25)
        time.sleep(sleep_time)

        X = np.concatenate([np.load(dtask["X"].path) for dtask in self.input()["data"]], axis=0)

        sleep_time = np.random.choice([1, 2, 3, 5])
        self.set_status_message(f"Loading responses!")
        self.set_progress_percentage(50)
        time.sleep(sleep_time)

        y = np.concatenate([np.load(dtask["y"].path) for dtask in self.input()["data"]], axis=0)

        sleep_time = np.random.choice([1, 2, 3, 5])
        self.set_status_message(f"Loading parameters!")
        self.set_progress_percentage(75)
        time.sleep(sleep_time)

        dtrain = xgb.DMatrix(X, label=y.squeeze())
        with self.input()["config"].open("r") as f:
            param = json.load(f)

        sleep_time = np.random.choice([10, 15, 20, 30])
        self.set_status_message(f"Training model with {param['num_rounds']} trees!")
        self.set_progress_percentage(90)
        time.sleep(sleep_time)
        bst = xgb.train(param, dtrain)
        bst.save_model(self.output().path)


class PredictTree(luigi.Task):
    fit_task = luigi.TaskParameter()
    data_task = luigi.TaskParameter()
    year = luigi.IntParameter()

    def requires(self):
        return {
            "model": self.fit_task,
            "data": self.data_task,
        }
    
    def output(self):
        model_dir = pjoin(scratch_dir, "forecasts")
        os.makedirs(model_dir, exist_ok=True)

        return luigi.LocalTarget(pjoin(model_dir, f"forecasts_{self.year}.npy"))

    def run(self):
        sleep_time = np.random.choice([3, 6, 9, 12])
        self.set_status_message(f"Loading data to predict!")
        self.set_progress_percentage(25)
        time.sleep(sleep_time)

        X_test = np.load(self.input()["data"]["X"].path)
        dtest = xgb.DMatrix(X_test)

        sleep_time = np.random.choice([1, 2, 3, 5])
        self.set_status_message(f"Loading model!")
        self.set_progress_percentage(50)
        time.sleep(sleep_time)
        
        bst = xgb.Booster({'nthread': 1})  # init model
        bst.load_model(self.input()["model"].path)
        y_hat = bst.predict(dtest)
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