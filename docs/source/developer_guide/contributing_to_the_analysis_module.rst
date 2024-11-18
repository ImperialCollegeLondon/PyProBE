.. _contributing_to_the_analysis_module:

Contributing to the Analysis Module
===================================

:mod:`pyprobe.analysis` classes are classes that perform further analysis of the data.

This document describes the standard format to be used for all PyProBE analysis functions. 
Constructing your method in this way ensures compatibility with the rest of the 
PyProBE package, while keeping your code clean and easy to read.


Functions
---------

All calculations should be conducted inside methods. These are called by the user with
any additional information required to perform the analysis, and always return 
:class:`~pyprobe.result.Result` objects. We will use the 
:func:`~pyprobe.analysis.differentiation.Differentiation.gradient` method as an example. 

It is recommended to use pydantic's `validate_call <https://docs.pydantic.dev/latest/api/validate_call/#pydantic.validate_call_decorator.validate_call>`_ 
function decorator to ensure that
objects of the correct type are being passed to your method. This provides the user with
an error message if they have not called the method correctly, simplifying debugging.

The steps to write a method are as follows:

1. Define the method and its input parameters. One of these is likely to be a PyProBE
   object, which you can confirm has the necessary columns for your method with step
   2.
2. Check that inputs to the method are valid with the 
   :class:`~pyprobe.analysis.utils.AnalysisValidator` class. Provide the class the 
   input data to the method, the columns that are required for the computation to 
   be performed and the required data type for `input_data``.
3. If needed, you can retrieve the columns specified in the `required_columns` field
   as numpy arrays by accessing the :attr:`~pyprobe.analysis.utils.AnalysisValidator.variables`
   attribute of the instance of :class:`~pyprobe.analysis.utils.AnalysisValidator`.
4. Perform the required computation. In this example, this is done with :func:`np.gradient`,
   a numpy built-in method. It is encouraged to perform as little of the underlying
   computation as possible in the analysis class method. Instead, write simple functions
   in the :mod:`pyprobe.analysis.base` module that process only numpy arrays. This
   keeps the mathematical underpinnings of PyProBE analysis methods readable, portable and
   testable.
5. Create a result object to return. This is easily done with the :func:`~pyprobe.result.Result.clean_copy`
   method, which provides a copy of the input data including the `info` attribute but
   replacing the data stored with a dataframe created from the provided dictionary.
6. Add column definitions to the created result object.
7. Return the result object.

.. literalinclude:: ../../../pyprobe/analysis/differentiation.py
    :language: python
    :linenos:
    :pyobject: gradient

Base
----

The :mod:`pyprobe.analysis.base` module exists as a repository for functions to work in
the rest of the analysis module. Often with data analysis code, it is tempting to include
data manipulation (forming arrays, dataframes etc. from your standard data format) 
alongside calculations. By keeping the data manipulation inside the methods of classes
in the :mod:`pyprobe.analysis` and calculations in the :mod:`~pyprobe.analysis.base`
submodule, these functions remain more readable, testable and portable.

:mod:`~pyprobe.analysis.base` module functions should be defined as simply as possible, 
accepting and returning only arrays and floating-point numbers, with clearly defined
variables. A good example is the
:func:`~pyprobe.analysis.base.degradation_mode_analysis_functions.calc_electrode_capacities` function
in the :mod:`~pyprobe.analysis.base.degradation_mode_analysis_functions` module:

.. literalinclude:: ../../../pyprobe/analysis/base/degradation_mode_analysis_functions.py
    :language: python
    :linenos:
    :pyobject: calc_electrode_capacities

.. footbibliography::