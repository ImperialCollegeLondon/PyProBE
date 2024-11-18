Analysis
========

Once filtered, the PyProBE :mod:`~pyprobe.analysis` module exists to perform further computation on your 
experimental data. You can see the currently available methods in this part of the API 
documentation.

Analysis modules contain functions and classes that operate on PyProBE objects.
Analysis classes and functions can be selective about the data that you provide to them. For example,
the :class:`~pyprobe.analysis.cycling.Cycling` analysis class requires the :code:`input_data`
attribute to be assigned an :class:`~pyprobe.filters.Experiment` object. This is to
allow this class to use attributes such as :func:`~pyprobe.filters.Experiment.charge`
internally. PyProBE will provide an error if the incorrect type is provided.

An analysis function may be dependent on specific columns in your experimental data. This
is validated when an analysis method is called, and an error is 
provided if the validation is not passed.

Most analysis functions are available at the module-level. In general:

.. code-block:: python

   result = analysis_modue.method(method_parameters)

or for the performing differentiation using the :func:`pyprobe.analysis.differentiation.gradient` function:

.. code-block:: python

   from pyprobe.analysis import differentiation
   gradient = differentiation.gradient(input_data = input_data,
                                       x = "Capacity [Ah]",
                                       y = "Voltage [V]")

Methods within analysis modules and classes always return :class:`~pyprobe.result.Result` objects, 
which allows direct integration with other PyProBE functionality such as plotting, and
other methods.

Analysis functions have been designed to be simple to read and implement. See the 
:ref:`Contributing to the analysis module <contributing_to_the_analysis_module>` section of the 
:ref:`Developer Guide <developer_guide>`. 

.. footbibliography::