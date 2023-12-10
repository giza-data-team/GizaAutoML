import time
from GizaAutoML.meta_model.optimization.tune_hyperparameters import HyperparameterOptimizerBO
import signal
from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design import DefaultInitialDesign
from math import ceil
from GizaAutoML.enums.regression_algorithms_enum import RegressionAlgorithmsEnum
from GizaAutoML.data_modeling.regressors import Regressors
from GizaAutoML.enums.algorithms_mapping import AlgorithmsMappingEnum


class MetaModelUtils:

    def __init__(self, time_stamp_col_name, target_col_name, random_seed):
        self.time_stamp_col_name = time_stamp_col_name
        self.target_col_name = target_col_name
        self.random_seed = random_seed

    def get_algorithm_search_space(self, top_3_algorithms):
        """
        Get the search space for the top-performing algorithms.

        This method extracts the search space (hyperparameter configurations) for the top-performing
        algorithms based on their names and defaults.

        Args:
            top_3_algorithms (DataFrame): A DataFrame containing information about the top-performing
                                          regression algorithms.

        Returns:
            algorithms_info (dict): A dictionary containing algorithm names as keys and their respective
                                    search space information as values.
        """
        top_3_algorithms_names = list(top_3_algorithms['algorithm'])
        print(top_3_algorithms_names)

        algorithms_info = {}
        for algorithm in top_3_algorithms_names:
            best_algorithm_instance = top_3_algorithms[top_3_algorithms['algorithm'] == algorithm]

            best_algorithm = RegressionAlgorithmsEnum[algorithm]
            regressor = Regressors(regressor_name=best_algorithm,
                                   label_col_name=self.target_col_name,
                                   time_stamp_col_name=self.time_stamp_col_name)
            search_space = regressor.get_regressor_params(smac_optimization=True,
                                                          random_seed=self.random_seed)

            defaults = best_algorithm_instance['hyperparameters'].iloc[0]
            algorithms_info[algorithm] = {'algorithm_instance': regressor,
                                          'search_space': search_space,
                                          'defaults': defaults}
        return algorithms_info

    @staticmethod
    def hyper_opt(x, y, algorithms, configurations, default_configurations, time_budget,
                  random_seed):
        """
        Perform hyperparameter optimization for multiple algorithms.

        This method performs hyperparameter optimization using Bayesian Optimization (SMAC) for multiple
        algorithms and returns the best hyperparameters and associated information for each algorithm.

        Args:
            x (DataFrame): The feature data for training.
            y (Series): The target data for training.
            algorithms (list): A list of algorithm names.
            configurations (list): A list of configuration spaces for the algorithms.
            default_configurations (list): A list of default hyperparameter configurations.
            time_budget (int): The time budget (in minutes) for training the meta-model.
            random_seed (int): Random seed for reproducibility.

        Returns:
            meta_models_info (dict): A dictionary containing information about the hyperparameter optimization
                                     results for each algorithm.
        """
        # divide time budget on the number of algorithms
        time_budgets = time_budget / len(algorithms)

        meta_models_info = {}
        for i, cs in enumerate(configurations):
            optimizer = HyperparameterOptimizerBO(cs, x, y,
                                                  algorithm_name=AlgorithmsMappingEnum[algorithms[i]].value,
                                                  random_seed=random_seed)

            scenario = Scenario(cs, n_trials=10000000, walltime_limit=time_budgets * 60)
            initial_design = DefaultInitialDesign(scenario)
            smac = HyperparameterOptimizationFacade(scenario, optimizer.train,
                                                    initial_design=initial_design,
                                                    overwrite=True)

            start = time.time()

            def handler(signum, frame):
                raise TimeoutError("Optimization timed out")

            # Set the maximum allowed time in seconds
            if ceil(time_budgets) > 1:
                time_budget = ceil(time_budgets) - 1
            else:
                time_budget = ceil(time_budgets)

            max_time_seconds = time_budget * 60  # Set your desired maximum time
            # Register the signal handler
            signal.signal(signal.SIGALRM, handler)

            try:
                # Set an alarm to trigger the handler after max_time_seconds
                signal.alarm(max_time_seconds)

                # Run the optimization
                incumbent = smac.optimize()

                # Disable the alarm as the optimization executed within the time limit
                signal.alarm(0)
            except TimeoutError:
                print("Optimization took too long. Terminating.")
                # Handle the timeout as needed
            finally:
                # Ensure the alarm is disabled even if an exception occurs
                signal.alarm(0)

            print("time:", time.time() - start)
            # Access the run history from the SMAC object

            runhistory = smac.runhistory.get_configs(sort_by='cost')
            print(runhistory)
            best_params = runhistory[0]
            print(best_params)
            min_cost = smac.runhistory.get_cost(best_params)
            print(f"min cost:,{min_cost}")

            run_id = smac.runhistory.get_config_id(best_params)
            print(run_id)

            best_hyperparameters = dict(best_params)
            print(f"best params:{dict(best_params)}")
            meta_models_info[algorithms[i]] = {'min_cost': min_cost,
                                               'hyperparameters': best_hyperparameters,
                                               'time': time.time() - start,
                                               'default': default_configurations[i],
                                               'trial_1': smac.runhistory.get_configs()[0],
                                               'trials_no': len(runhistory)}
        return meta_models_info

