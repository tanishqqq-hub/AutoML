"""
Preprocessing module for AutoML Studio.

Builds and fits a sklearn Pipeline that handles numerical
and categorical transformations without data leakage.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
)
from sklearn.model_selection import train_test_split


class Preprocessor:
    """
    Handles preprocessing for AutoML Studio.

    Responsibilities
    ---------------
    - Separate features and target
    - Build preprocessing pipeline
    - Train/test split
    - Fit transformations
    - Persist preprocessing pipeline artifact
    """

    def __init__(self, config, logger, artifact_manager):
        self.config = config
        self.logger = logger
        self.artifact_manager = artifact_manager
        self.target_column = self.config.data.target_column

    def _split_features_target(
        self, df: pd.DataFrame, task_type: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separate features (X) and target (y).

        Parameters
        ----------
        df : pd.DataFrame
            Validated dataset.
        task_type : str
            Detected task type — "classification" or "regression".

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            Feature dataframe and target series.

        Raises
        ------
        ValueError
            If target column is not found in dataframe.
        """
        self.logger.info("Splitting features and target.")

        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataframe."
            )

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        self.logger.info(
            f"Feature matrix shape: {X.shape} | Target shape: {y.shape}"
        )

        if task_type == "classification":
            self.logger.info(f"Target distribution:\n{y.value_counts()}")
        else:
            self.logger.info(
                f"Target stats — min: {y.min():.4f} | "
                f"max: {y.max():.4f} | mean: {y.mean():.4f}"
            )

        return X, y

    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """
        Build an unfitted preprocessing pipeline.

        Detects numerical and categorical columns automatically
        and applies appropriate transformations to each.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature dataframe.

        Returns
        -------
        Pipeline
            Unfitted sklearn Pipeline.

        Raises
        ------
        ValueError
            If an unsupported scaling method is configured.
        """
        self.logger.info("Building preprocessing pipeline.")

        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        categorical_cols = X.select_dtypes(include="object").columns.tolist()

        self.logger.info(
            f"Detected {len(numeric_cols)} numerical columns "
            f"and {len(categorical_cols)} categorical columns."
        )

        scaler_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }

        scaling_method = self.config.preprocessing.scaling

        if scaling_method not in scaler_map:
            raise ValueError(
                f"Unsupported scaling method '{scaling_method}'. "
                f"Available options: {list(scaler_map.keys())}"
            )

        scaler = scaler_map[scaling_method]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", scaler),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, categorical_cols),
            ]
        )

        pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

        self.logger.info("Preprocessing pipeline successfully constructed.")
        return pipeline

    def run(
        self,
        df: pd.DataFrame,
        task_type: str,
        experiment_id: str,
    ) -> tuple:
        """
        Execute the full preprocessing workflow.

        Parameters
        ----------
        df : pd.DataFrame
            Validated dataset from the validation stage.
        task_type : str
            Detected task type — "classification" or "regression".
        experiment_id : str
            Unique experiment identifier for artifact storage.

        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test as numpy arrays.
        """
        self.logger.info("--- Preprocessing Started ---")

        X, y = self._split_features_target(df, task_type)

        test_size = self.config.data.test_size
        random_state = self.config.data.random_seed

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if task_type == "classification" else None,
        )

        self.logger.info(
            f"Dataset split — Train: {X_train.shape[0]} rows | "
            f"Test: {X_test.shape[0]} rows"
        )

        pipeline = self._build_pipeline(X_train)

        self.logger.info("Fitting preprocessing pipeline on training data.")
        pipeline.fit(X_train)

        self.logger.info("Transforming training and test datasets.")
        X_train = pipeline.transform(X_train)
        X_test = pipeline.transform(X_test)

        self.logger.info("Saving preprocessing pipeline artifact.")
        self.artifact_manager.save_model(
            pipeline,
            "preprocessing_pipeline",
            experiment_id,
        )

        self.logger.info(
            f"--- Preprocessing Complete --- | "
            f"X_train: {X_train.shape} | X_test: {X_test.shape} | "
            f"y_train: {y_train.shape} | y_test: {y_test.shape}"
        )

        return X_train, X_test, y_train, y_test