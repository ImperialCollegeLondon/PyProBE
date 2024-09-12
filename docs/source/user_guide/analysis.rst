Analysis
========

Once filtered, the PyProBE :mod:`~pyprobe.analysis` exists to perform further computation on your 
experimental data. You can see the currently available methods in this part of the API 
documentation.

An analysis class must be instantiated by providing a variant of a 
:class:`~pyprobe.result.Result` object to the :code:`input_data` field. This contains the data 
that you wish to perform the analysis on:

.. code-block:: python

   analysis_object = AnalysisClass(input_data = Result)

for example, to perform :mod:`degradation mode analysis <pyprobe.analysis.degradation_mode_analysis>` on a discharge pseudo-ocv:

.. code-block:: python

   discharge_pOCV = cell.procedure['Procedure Name'].experiment('pOCV').discharge(0)
   dma_object = DMA(input_data = discharge_pOCV)

Analysis classes can be selective about the data that you provide to them. For example,
the :class:`~pyprobe.analysis.cycling.Cycling` analysis class requires the :code:`input_data`
attribute to be assigned an :class:`~pyprobe.filters.Experiment` object. This is to
allow this class to use attributes such as :func:`~pyprobe.filters.Experiment.charge`
internally. PyProBE will provide an error if the incorrect type is provided.

An analysis class may be dependent on specific columns in your experimental data. This
is also validated when an instance of an analysis class is created, and an error is 
provided if the validation is not passed.

To perform the analysis, you must then call a method that is an attribute of this class.
In the call to the method, you must provide any additional parameters needed for the 
method. In general:

.. code-block:: python

   result = analysis_object.method(method_parameters)

or for the DMA example, using the :func:`~pyprobe.analysis.degradation_mode_analysis.DMA.fit_ocv` method:

.. code-block:: python

   stoichiometry_limits = dma_object.fit_ocv(x_ne, x_pe, ocp_ne, ocp_pe, x_guess)

Methods within analysis classes always return :class:`~pyprobe.result.Result` objects, 
which allows direct integration with other PyProBE functionality such as plotting, and
other methods.

Analysis functions have been designed to be simple to read and implement. See the 
:ref:`Contributing to the analysis module <contributing_to_the_analysis_module>` section of the 
:ref:`Developer Guide <developer_guide>`. 

.. footbibliography::