from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition, OrConjunction


class ConfigSpaceBuilder:
    def __init__(self, random_seed):
        self.random_seed = random_seed

    def build_config_space(self, search_spaces, defaults_list=None):
        config_spaces = []
        for search_space, defaults in zip(search_spaces, defaults_list):
            cs = ConfigurationSpace(seed=self.random_seed)
            for param, values in search_space.items():
                if isinstance(values[0], int) and not isinstance(values[0], bool):
                    min_val = min(values)
                    max_val = max(values)
                    default_val = defaults.get(param, min_val) if defaults else min_val
                    hyperparam = Integer(param, (min_val, max_val), default=default_val, log=True)
                elif isinstance(values[0], float):
                    min_val = min(values)
                    max_val = max(values)
                    default_val = defaults.get(param, min_val) if defaults else min_val
                    hyperparam = Float(param, (min_val, max_val), default=default_val, log=True)
                else:
                    default_val = defaults.get(param, values[0]) if defaults else values[0]
                    hyperparam = Categorical(param, values, default=default_val)

                cs.add_hyperparameter(hyperparam)

            config_spaces.append(cs)

        return config_spaces

    def current_step_conf_space(self):
        # Define hyperparameters for RandomForest
        n_estimators_rf = UniformIntegerHyperparameter(name='n_estimators', lower=100, upper=400)
        max_depth_rf = UniformIntegerHyperparameter(name='max_depth', lower=5, upper=40)

        # Define hyperparameters for LassoRegression
        alpha_lasso = UniformFloatHyperparameter(name='alpha', lower=0.00001, upper=0.3)

        # Define hyperparameters for AdaBoost
        n_estimators_adaboost = UniformIntegerHyperparameter(name='n_estimators', lower=50, upper=200)
        learning_rate_adaboost = UniformFloatHyperparameter(name='learning_rate', lower=0.01, upper=1.0)
        loss_adaboost = Categorical(name='loss', items= ['linear', 'square', 'exponential'])


        # Create the configuration space for each algorithm
        cs_rf = ConfigurationSpace(seed=self.random_seed)
        cs_rf.add_hyperparameters([n_estimators_rf, max_depth_rf])

        cs_lasso = ConfigurationSpace(seed=self.random_seed)
        cs_lasso.add_hyperparameters([alpha_lasso])

        cs_adaboost = ConfigurationSpace(seed=self.random_seed)
        cs_adaboost.add_hyperparameters([n_estimators_adaboost, learning_rate_adaboost, loss_adaboost])

        return [cs_rf, cs_lasso, cs_adaboost]

#     # Define multiple search spaces and defaults
# search_space1 = {
#     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
#     'C': [1, 2, 3, 5, 10, 15, 30],
#     'epsilon': [0.01, 0.05, 0.1, 0.2, 0.3]
# }
#
# defaults1 = {
#     'C': 15,
#     'kernel': 'rbf',
#     'epsilon': 0.01
# }
#
# search_space2 = {
#     'learning_rate': [0.001, 0.01, 0.1, 0.5],
#     'n_estimators': [50, 100, 200, 500],
#     'max_depth': [3, 5, 7, 10]
# }
#
# defaults2 = {
#     'learning_rate': 0.1,
#     'n_estimators': 200,
#     'max_depth': 5
# }
#
# # Create an instance of ConfigSpaceBuilder
# config_space_builder = ConfigSpaceBuilder()
#
# # Build ConfigurationSpace objects for each input set
# config_spaces = config_space_builder.build_config_space([search_space1, search_space2], [defaults1, defaults2])