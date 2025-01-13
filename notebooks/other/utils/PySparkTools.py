from pyspark.sql import DataFrame, Column, Window
from pyspark.sql.functions import sum as sum_, avg, max as max_, min as min_, percentile_approx, col, when, count

from dataclasses import dataclass
from typing import Union, List

@dataclass
class ProcessColumns:
    @staticmethod
    def get_aggregated_data(df: DataFrame, column_name: str, groupby_cols: list[str], operation: str) -> DataFrame:       
        valid_operations = {
            "sum": sum_,
            "avg": avg,
            "max": max_,
            "min": min_,
            "median": lambda col: percentile_approx(col, 0.5)
        }

        if operation not in valid_operations:
            raise ValueError(f"Unsupported operation '{operation}'. Valid options are: {', '.join(valid_operations.keys())}")

        agg_func = valid_operations[operation]
        return df.groupBy(groupby_cols).agg(agg_func(column_name).alias(f"{column_name}_{operation}"))
    
    @staticmethod
    def get_aggregated_data_multi_ops(
        df: DataFrame,
        column_name: str,
        groupby_cols: list[str],
        operations: list[str],
    ) -> DataFrame:
        """
        Get the aggregated data for multiple operations at once.

        Args:
            df (DataFrame): The DataFrame to process.
            column_name (str): The column to apply the operations on.
            groupby_cols (list[str]): The columns to group by.
            operations (list[str]): A list of operations to apply 
                                    (e.g., ['sum', 'avg', 'max', 'min', 'median']).

        Returns:
            DataFrame: A DataFrame with each requested operation as a separate column.
        """
        # Diccionario de operaciones válidas
        valid_operations = {
            "sum": sum_,
            "avg": avg,
            "max": max_,
            "min": min_,
            "median": lambda col: percentile_approx(col, 0.5),
        }
        
        # Construir dinámicamente las columnas de agregación
        agg_exprs = []
        for op in operations:
            if op not in valid_operations:
                raise ValueError(
                    f"Unsupported operation '{op}'. "
                    f"Valid options are: {', '.join(valid_operations.keys())}"
                )
            agg_exprs.append(
                valid_operations[op](column_name).alias(f"{column_name}_{op}")
            )

        # Realizar la agregación usando todas las funciones en una sola llamada
        return df.groupBy(groupby_cols).agg(*agg_exprs)

    @staticmethod
    def get_threshold_result(
        process_col: str,
        threshold: float,
        criteria: str = ">",
    ) -> Column:
        """
        Generate a threshold condition for a given column.

        Args:
            process_col (str): Name of the column to evaluate.
            threshold (float): Threshold value to compare against.
            criteria (str): Comparison operator (e.g., '>', '<=', '==', '!=', '>=', '<=').

        Returns:
            Column: A conditional column (True/None) based on the threshold logic.

        Raises:
            ValueError: If the provided criteria is not supported.
        """
        operators = {
            ">": lambda: col(process_col) > threshold,
            "<": lambda: col(process_col) < threshold,
            "==": lambda: col(process_col) == threshold,
            "!=": lambda: col(process_col) != threshold,
            ">=": lambda: col(process_col) >= threshold,
            "<=": lambda: col(process_col) <= threshold,
        }

        if criteria not in operators:
            raise ValueError(f"Unsupported criteria: {criteria}")

        condition = operators[criteria]()
        return when(condition, True).otherwise(None)

@dataclass
class WindowThresholdProcessor:
    """
    Processor for applying threshold-based logic on columns with window functions.
    """

    @staticmethod
    def process_column_uun(
        process_col: Column,
        n_months: int,
        partition_by_col: str,
        order_by_col: str,
        u_months: int,
        logs: bool = False,
    ) -> List[Column]:
        """
        Count how many rows match a given process_col condition over a window,
        then return a list of columns indicating if a threshold (u_months) is met.

        Args:
            process_col (Column): A Column object with a True/None condition.
            n_months (int): Number of rows to look ahead in the window.
            partition_by_col (str): Column by which to partition the data.
            order_by_col (str): Column by which to order the data (descending).
            u_months (int): Minimum number of True values required in the window.
            logs (bool): Whether to return extra debugging columns.

        Returns:
            List[Column]:
                - If logs=False, returns [is_unn].
                - If logs=True, returns [MonthsMet, is_unn].
        """
        window_spec = (
            Window.partitionBy(partition_by_col)
            .orderBy(col(order_by_col).desc())
            .rowsBetween(0, n_months)
        )

        count_months = count(process_col).over(window_spec)
        is_unn = when(count_months >= u_months, True).otherwise(None).alias("is_unn")

        if logs:
            months_met = count_months.alias("MonthsMet")
            return [months_met, is_unn]
        return [is_unn]

    @staticmethod
    def process_column_uun_with_threshold(
        process_col: str,
        threshold: float,
        criteria: str,
        n_months: int,
        partition_by_col: str,
        order_by_col: str,
        u_months: int,
        logs: bool = False,
    ) -> List[Column]:
        """
        In a single step, applies a threshold check to the given column and
        counts how many times that threshold is met over a window of n_months - 1.

        1. Applies get_threshold_result to generate the threshold condition.
        2. Adjusts n_months = n_months - 1.
        3. Calls process_column_uun using the column result.

        Args:
            process_col (str): Name of the original column to evaluate.
            threshold (float): Threshold value to compare against.
            criteria (str): Comparison operator (e.g., '>', '<=', '==', '!=', '>=', '<=').
            n_months (int): Number of rows to look ahead in the window.
            partition_by_col (str): Column by which to partition the data.
            order_by_col (str): Column by which to order the data (descending).
            u_months (int): Minimum number of True values required in the window.
            logs (bool): Whether to return extra debugging columns.

        Returns:
            List[Column]:
                - If logs=False, returns [is_unn].
                - If logs=True, returns  [MonthsMet, is_unn].
        """
        threshold_col = ProcessColumns.get_threshold_result(
            process_col=process_col,
            threshold=threshold,
            criteria=criteria
        )

        n_months = n_months - 1

        return WindowThresholdProcessor.process_column_uun(
            process_col=threshold_col,
            n_months=n_months,
            partition_by_col=partition_by_col,
            order_by_col=order_by_col,
            u_months=u_months,
            logs=logs,
        )


    def apply_uun_thresholds_multiple_cols(
        columns_info: list,
        n_months: int,
        partition_by_col: str,
        order_by_col: str,
        u_months: int,
        logs: bool = False,
    ):
        """
        Aplica 'process_column_uun_with_threshold' a múltiples columnas que
        comparten los mismos parámetros de ventana y de conteo.
        
        columns_info (list[dict]): 
            Lista de diccionarios, donde cada dict contiene:
            - 'col': str -> nombre de la columna
            - 'threshold': float -> valor umbral
            - 'criteria': str -> operador (e.g. ">", "<=", ...)
            
        Retorna:
            List[Column]: la concatenación de todas las columnas generadas 
                        para cada col en 'columns_info'.
        """
        all_new_cols = []
        for info in columns_info:
            new_cols = WindowThresholdProcessor.process_column_uun_with_threshold(
                process_col=info["col"],
                threshold=info["threshold"],
                criteria=info["criteria"],
                n_months=n_months,
                partition_by_col=partition_by_col,
                order_by_col=order_by_col,
                u_months=u_months,
                logs=logs
            )
            # 'process_column_uun_with_threshold' devuelve siempre una lista de Column
            # con 1 o 2 columnas según logs
            all_new_cols.extend(new_cols)
        
        return all_new_cols
    
@dataclass
class WindowThresholdBatchProcessor:
    """
    Holds window parameters and logic to be applied to multiple columns.

    Attributes:
        n_months (int): Number of rows to include in the window.
        partition_by_col (str): Column by which to partition the data.
        order_by_col (str): Column by which to order the data (descending).
        u_months (int): Minimum number of True values required in the window.
        logs (bool): Determines if additional debugging columns are returned.
    """
    n_months: int
    partition_by_col: str
    order_by_col: str
    u_months: int
    logs: bool = False

    def process(
        self, 
        process_col: str, 
        threshold: float, 
        criteria: str
    ) -> list[Column]:
        """
        Applies the method process_column_uun_with_threshold
        using the attributes defined in this dataclass as default parameters.
        Returns a list of Column objects (one or two, depending on logs).

        Args:
            process_col (str): Name of the column to evaluate.
            threshold (float): Threshold value for comparison.
            criteria (str): Comparison operator (e.g. '>', '<=', '==', '!=').

        Returns:
            list[Column]: A list of Column objects (one or two, depending on logs).
        """
        return WindowThresholdProcessor.process_column_uun_with_threshold(
            process_col=process_col,
            threshold=threshold,
            criteria=criteria,
            n_months=self.n_months,
            partition_by_col=self.partition_by_col,
            order_by_col=self.order_by_col,
            u_months=self.u_months,
            logs=self.logs
        )