Dependency Guide
================

Polars
------

`Polars <https://docs.pola.rs/api/python/stable/reference/index.html>`_ is the most 
important dependency in PyProBE. It provides all of the back-end
data storage and manipulation through LazyFrames and DataFrames. The principle of 
Lazy evaluation is critical to the performance of PyProBE. The polars 
`expression <https://docs.pola.rs/user-guide/expressions/>`_ 
framework is used throughout PyProBE as it allows multiple commands to be chained 
together. PyProBE constructs this chain of commands as the user filters the data, which
is only executed when the final result is requested from the user. This might be to
print, plot or perform analysis on the data with the `analysis` module.

The performance of PyProBE is demonstrated in the :doc:`../examples/comparing-pyprobe-performance` example.

Pydantic
--------

`Pydantic <https://docs.pydantic.dev/latest/>`_ is used across PyProBE for class and function input
validation. :class:`~pyprobe.result.Result`, :class:`~pyprobe.rawdata.RawData` and 
all of the classes in the :mod:`~pyprobe.filters` module inherit from Pydantic 
`BaseModel <https://docs.pydantic.dev/latest/api/base_model/>`_. This means all of their
inputs are type-validated automatically. 

Functions in the :mod:`~pyprobe.analysis` module use the `validate_call <https://docs.pydantic.dev/latest/api/validate_call/#pydantic.validate_call_decorator.validate_call>`_ 
decorator, to ensure the arguments passed are of the correct type.
The :class:`~pyprobe.analysis.utils.AnalysisValidator`
is a custom validation model for checking the type and columns are correct for methods
of analysis classes.

Most of this work is behind-the-scenes. Your main interaction with Pydantic is likely to
be when contributing to the analysis module, so follow this :ref:`guide <contributing_to_the_analysis_module>`
to set it up correctly.

Examples of the causes of Pydantic validation errors can be found in the :doc:`../examples/providing-valid-inputs`.
example.


.. footbibliography::