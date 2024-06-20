import abc
import typing as t

import pandas as pd

from pyspark.sql import SparkDataFrame


class BaseFeatureTransformer(abc.ABC):

    @t.overload
    @abc.abstractmethod
    def transform(self) -> SparkDataFrame: ...

    @t.overload
    @abc.abstractmethod
    def transform(self, df: SparkDataFrame) -> SparkDataFrame: ...

    @t.overload
    @abc.abstractmethod
    def transform(
        self, df: SparkDataFrame, label_month_start_date: str, label_month_end_date: str
    ) -> SparkDataFrame: ...

    def create_label_month_list(
        self, label_month_start_date: str, label_month_end_date: str
    ) -> list[str]:
        return [
            d.strftime("%Y%m")
            for d in pd.date_range(
                label_month_start_date, label_month_end_date, freq="MS"
            )
        ]

    def _join(
        self, feature_table: SparkDataFrame, new_table: SparkDataFrame
    ) -> SparkDataFrame:
        return feature_table.join(
            new_table, on=["Master_Account_Key", "label_month"], how="left"
        )


class FeatureAggregator: ...
