"""
Data validation module for AutoML Studio.

Acts as a quality gate before preprocessing. Validates
null thresholds, duplicates, target integrity, and feature variance.
"""

import logging
from types import SimpleNamespace

import pandas as pd


class DataValidator:
    """
    Validates the raw dataset before it enters the preprocessing pipeline.

    Parameters
    ----------
    config : SimpleNamespace
        Loaded configuration object.
    logger : logging.Logger
        Configured logger instance.
    """

    def __init__(self, config: SimpleNamespace, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute all validation checks on the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataset from ingestion stage.

        Returns
        -------
        pd.DataFrame
            Validated and deduplicated dataset.

        Raises
        ------
        ValueError
            If any fatal validation check fails.
        """
        self.logger.info("--- Data Validation Started ---")

        self._check_min_rows(df)
        self._check_target(df)
        df=self._check_nulls(df)
        df = self._check_duplicates(df)
        self._check_feature_variance(df)

        self.logger.info("--- Data Validation Passed ---")
        return df

    def _check_min_rows(self, df: pd.DataFrame) -> None:
        """
        Validate the dataset meets the minimum row requirement.

        Raises
        ------
        ValueError
            If the dataset has fewer rows than configured minimum.
        """
        min_rows = self.config.validation.min_rows
        if len(df) < min_rows:
            raise ValueError(
                f"Dataset has {len(df)} rows, which is below the "
                f"minimum required: {min_rows}."
            )
        self.logger.info(f"Row count check passed — {len(df)} rows found.")

    def _check_target(self, df: pd.DataFrame) -> None:
        """
        Validate the target column exists and has sufficient variability.

        Raises
        ------
        ValueError
            If the target column is missing or has insufficient variability.
        """
        target_column = self.config.data.target_column

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataset."
            )

        unique_values = df[target_column].nunique()
        if unique_values < 2:
            raise ValueError(
                f"Target column '{target_column}' must contain at least "
                f"2 unique values. Found {unique_values}."
            )

        self.logger.info(
            f"Target validation passed — column '{target_column}' "
            f"contains {unique_values} unique values."
        )

    def _check_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check null percentage per column and drop columns exceeding threshold.

        Columns exceeding the configured null threshold are automatically dropped,
        except for the target column which will trigger a fatal error.

        Returns
        -------
        pd.DataFrame
            Dataframe with high-null columns removed.

        Raises
        ------
        ValueError
            If the target column exceeds the null threshold.
        """

        threshold = self.config.validation.null_threshold
        target_column = self.config.data.target_column

        total_nulls = df.isnull().sum().sum()
        self.logger.info(f"Total null values found in dataset: {total_nulls}")

        null_percent = df.isnull().mean()

        drop_columns = []

        for col, pct in null_percent.items():

            if pct > threshold:

                if col == target_column:
                    raise ValueError(
                        f"Target column '{target_column}' exceeds null threshold "
                        f"({pct:.2%} > {threshold:.2%}). Cannot continue."
                    )

                self.logger.warning(
                    f"Column '{col}' has {pct:.2%} null values "
                    f"(threshold={threshold:.2%}). Marking for removal."
                )

                drop_columns.append(col)

        if drop_columns:
            df = df.drop(columns=drop_columns)

            self.logger.warning(
                f"Dropped columns due to excessive null values: {', '.join(drop_columns)}"
            )

        else:
            self.logger.info("No columns exceed the null threshold.")

        return df
    
    def _check_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and remove duplicate rows.

        Returns
        -------
        pd.DataFrame
            Dataframe with duplicates removed.
        """
        duplicate_count = df.duplicated().sum()

        if duplicate_count == 0:
            self.logger.info("No duplicate rows detected.")
            return df

        self.logger.warning(f"Duplicate rows detected: {duplicate_count}")
        df_cleaned = df.drop_duplicates()
        self.logger.info(
            f"Duplicates removed. Remaining rows: {len(df_cleaned)}"
        )
        return df_cleaned

    def _check_feature_variance(self, df: pd.DataFrame) -> None:
        """
        Identify numeric features with zero variance.

        Zero variance columns carry no information for model training.
        Logged as a warning — does not stop the pipeline.
        """
        target_column = self.config.data.target_column
        numeric_cols = df.select_dtypes(include="number").columns
        numeric_cols = [col for col in numeric_cols if col != target_column]

        zero_var_cols = [
            col for col in numeric_cols if df[col].std() == 0
        ]

        if zero_var_cols:
            self.logger.warning(
                f"Zero variance detected in columns: {', '.join(zero_var_cols)}"
            )
        else:
            self.logger.info("All numeric features have non-zero variance.")