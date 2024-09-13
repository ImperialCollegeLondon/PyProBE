.. _contributing_to_the_analysis_module:

Contributing to the Analysis Module
===================================

:mod:`pyprobe.analysis` classes are classes that perform further analysis of the data.

This document describes the standard format to be used for all PyProBE analysis classes. 
Constructing your method in this way ensures compatibility with the rest of the 
PyProBE package, while keeping your code clean and easy to read.

Analysis classes are based on `Pydantic BaseModel <https://docs.pydantic.dev/latest/api/base_model/>`_ 
to provide input validation. However, following the steps below should allow
you to write your own analysis class without any direct interaction with pydantic 
itself. 

Setup
-----
1. Start by creating your class, which must inherit from 
   pydantic :code:`BaseModel`.
2. Declare :code:`input_data` as a variable and specify its type. The :mod:`pyprobe.typing`
   module has type aliases that may be helpful here. This type should be the most
   lenient type that the methods of your analysis class require.

.. literalinclude:: ../../../pyprobe/analysis/differentiation.py
    :language: python
    :linenos:
    :lines: 15-21

3. Some analysis classes have multiple methods that need to pass information to each 
   other. For instance the :class:`~pyprobe.analysis.degradation_mode_analysis.DMA`
   analysis class first calculates stoichiometry limits with the 
   :func:`~pyprobe.analysis.degradation_mode_analysis.DMA.fit_ocv` method, that are then
   used in the :func:`~pyprobe.analysis.degradation_mode_analysis.DMA.quantify_degradation_modes`
   method. So, when :func:`~pyprobe.analysis.degradation_mode_analysis.DMA.fit_ocv` is called, 
   it saves this result in `stoichiometry_limits` for use later. If they are required, 
   these attributes must also be defined at the top of the class.

.. literalinclude:: ../../../pyprobe/analysis/degradation_mode_analysis.py
    :language: python
    :linenos:
    :lines: 17-31


Then you can add any additional methods to perform your calculations. 

Methods
-------

All calculations should be conducted inside methods. These are called by the user with
any additional information required to perform the analysis, and always return 
:class:`~pyprobe.result.Result` objects. We will use the 
:func:`~pyprobe.analysis.differentiation.Differentiation.differentiate_FD` method as an example. 
The steps to write a method are as follows:

1. Define the method and its input parameters.
2. Check that inputs to the method are valid with the 
   :class:`~pyprobe.analysis.utils.AnalysisValidator` class. Provide the class the 
   input data to the method, the columns that are required for the computation to 
   be performed and the required data type for `input_data`` (only if it is a stricter 
   requirement than the type assigned to `input_data` above).
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
    :pyobject: Differentiation.differentiate_FD

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