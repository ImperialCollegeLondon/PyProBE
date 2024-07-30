Analysis
========

Once filtered, PyProBE analysis classes are there to perform further computation on your 
experimental data. They are contained within the :mod:`~pyprobe.analysis` 
module, and you can see the currently available methods in this part of the API 
documentation.

An analysis class must be instantiated with a variant of a 
:class:`~pyprobe.result.Result` object. This contains the data that you wish to perform
the analysis on. 

Analysis functions always return :class:`~pyprobe.result.Result` objects, which allows direct
integration with other PyProBE functionality such as plotting, or other methods.

Analysis functions have been designed to be simple to read and implement. See the 
:ref:`Contributing to the analysis module <contributing_to_the_analysis_module>` section of the 
:ref:`Developer Guide <developer_guide>`. 

.. footbibliography::