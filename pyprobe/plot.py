"""A module to contain plotting functions for PyProBE."""

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import polars as pl
from deprecated import deprecated
from numpy.typing import NDArray
from plotly.express.colors import sample_colorscale
from plotly.subplots import make_subplots
from sklearn.preprocessing import minmax_scale

if TYPE_CHECKING:
    from pyprobe.result import Result

from pyprobe.units import split_quantity_unit


def _retrieve_relevant_columns(
    result_obj: "Result", args: Tuple[Any, ...], kwargs: Dict[Any, Any]
) -> pl.DataFrame:
    """Retrieve relevant columns from a Result object for plotting.

    This function analyses the arguments passed to a plotting function and retrieves the
    used columns from the Result object.

    Args:
        result_obj: The Result object.
        args: The positional arguments passed to the plotting function.
        kwargs: The keyword arguments passed to the plotting function.

    Returns:
        A dataframe containing the relevant columns from the Result object.
    """
    kwargs_values = [
        v for k, v in kwargs.items() if isinstance(v, str) and k != "label"
    ]
    args_values = [v for v in args if isinstance(v, str)]
    all_args = set(kwargs_values + args_values)
    relevant_columns = []
    for arg in all_args:
        try:
            quantity, _ = split_quantity_unit(arg)

        except ValueError:
            continue
        if quantity in result_obj._polars_cache.quantities:
            relevant_columns.append(arg)
    if len(relevant_columns) == 0:
        raise ValueError(
            f"None of the columns in {all_args} are present in the Result object."
        )
    result_obj._polars_cache.collect_columns(*relevant_columns)
    return result_obj._get_data_subset(*relevant_columns)


try:
    import seaborn as _sns
except ImportError:
    _sns = None


def _create_seaborn_wrapper() -> Any:
    """Create a wrapped version of the seaborn package."""
    if _sns is None:

        class SeabornWrapper:
            def __getattr__(self, _: Any) -> None:
                """Raise an ImportError if seaborn is not installed."""
                raise ImportError(
                    "Optional dependency 'seaborn' is not installed. Please install by "
                    "running 'pip install seaborn' or installing PyProBE with seaborn "
                    "as an optional dependency: `pip install 'PyProBE-Data[seaborn]'."
                )

        return SeabornWrapper()

    wrapped_sns = type("SeabornWrapper", (), {})()

    def wrap_function(func: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a seaborn function.

        Args:
            func (Callable): The function to wrap.
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """The wrapper function.

            Modifies the 'data' argument to seaborn functions to be compatible with
            PyProBE Result objects.

            Args:
                *args: The positional arguments.
                **kwargs: The keyword arguments.

            Returns:
                The result of the wrapped function.
            """
            if "data" in kwargs:
                kwargs["data"] = _retrieve_relevant_columns(
                    kwargs["data"], args, kwargs
                ).to_pandas()
            if func.__name__ == "lineplot":
                if "estimator" not in kwargs:
                    kwargs["estimator"] = None
            return func(*args, **kwargs)

        return wrapper

    # Copy all seaborn attributes
    for attr_name in dir(_sns):
        if not attr_name.startswith("_"):
            attr = getattr(_sns, attr_name)
            if callable(attr):
                setattr(wrapped_sns, attr_name, wrap_function(attr))
            else:
                setattr(wrapped_sns, attr_name, attr)

    return wrapped_sns


seaborn = _create_seaborn_wrapper()
"""A wrapped version of the seaborn package.

Requires the seaborn package to be installed as an optional dependency. You can install
it with PyProBE by running :code:`pip install 'PyProBE-Data[seaborn]'`, or install it
seperately with :code:`pip install seaborn`.

This version of seaborn is modified to work with PyProBE Result objects. All functions
from the original seaborn package are available in this version. Where seaborn functions
accept a 'data' argument, a PyProBE Result object can be passed instead of a pandas
DataFrame. For example:

.. code-block:: python

    from pyprobe.plot import seaborn as sns

    result = cell.procedure['Sample']
    sns.lineplot(data=result, x="x", y="y")

Other modifications include:
    - The 'estimator' argument is set to None by default in the lineplot function for
    performance. This can be overridden by passing an estimator explicitly.

See the seaborn documentation for more information: https://seaborn.pydata.org/
"""
