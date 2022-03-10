from dataclasses import dataclass

import numpy as np
from configs.SklearnClassifierFlowConfig import SklearnClassifierFlowConfig
from data.utils import get_data, preprocess_data
from metaflow import FlowSpec, JSONType, Parameter, current, step
from model.utils import create_numpy_X_y_dataset_from_df, hyperparameter_search
from pytorch_lightning import seed_everything

 
def check_config(config: dataclass) -> dataclass:
    if "missing" in getattr(config, "default_hyperparameters") and "XGB" in getattr(config, "clf_type"):
        config.default_hyperparameters["missing"] = np.nan


class SklearnClassifierFlow(FlowSpec):

    # Override default configuration parameters with json configuration.
    config_overrides = Parameter(name="config_overrides", help="JSON model configuration", type=JSONType, default="{}")

    @step
    def start(self):
        self.next(self.sklearn_classifier)

    @step
    def sklearn_classifier(self):

        # Info.
        print("Instantiate configuration.")

        # Instantiate configuration.
        config = SklearnClassifierFlowConfig(**self.config_overrides)  # pylint: disable=not-a-mapping
        config.experiment_name = f"flow_name_{current.flow_name}_clf_type_{config.clf_type}_experiment_name_{config.experiment_name}_run_id_{current.run_id}"
        check_config(config)

        # Set seed for random number generators in pytorch, numpy, and python.random
        if config.seed >= 0:
            seed_everything(config.seed, workers=True)

        # Info.
        print("Getting and preprocess data.")

        # Get and preprocess data as a dataframe.
        df_preprocessed, labels, number_features, category_features = get_data(config.get_data_method)
        self.num_samples = len(df_preprocessed)
        config.labels = labels
        config.number_features = number_features
        config.category_features = category_features
        self.config = config.__dict__

        # Info.
        print(f"Running sklearn classifer flow using config: {config}.")

        # Create X,y numpy arrays from the dataframe and label encoders for categorical data.
        X, y, label_encoder_dicts = create_numpy_X_y_dataset_from_df(
            self.num_samples,
            df_preprocessed,
            config.labels,
            config.number_features,
            config.category_features,
            random_state=config.random_state,
        )
        self.label_encoder_dicts = label_encoder_dicts

        # Perform hyperparameter search and save search results.
        (
            results,
            best_val_accuracy,
            best_test_accuracy,
            best_hyperparameters,
            best_clf,
            best_scaler,
        ) = hyperparameter_search(
            X=X,
            y=y,
            default_hyperparameters=config.default_hyperparameters,
            clf_type=config.clf_type,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            permutations=config.permutations,
            hyperparameter_search_type=config.hyperparameter_search_type,
            test_size=config.test_size,
            val_size=config.val_size,
            hyperparameter_num_searches=config.hyperparameter_num_searches,
        )
        self.results = results
        self.best_val_accuracy = best_val_accuracy
        self.best_test_accuracy = best_test_accuracy
        self.best_hyperparameters = best_hyperparameters
        self.best_clf = best_clf
        self.best_scaler = best_scaler

        # End training.
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    SklearnClassifierFlow()
