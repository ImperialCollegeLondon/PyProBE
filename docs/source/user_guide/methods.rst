Analysis
========

Once filtered, PyProBE analysis functions are there to perform further computation on your 
experimental data. They are contained within the :mod:`analysis <~pyprobe.analysis>` 
module, and you can see the currently available methods in this part of the API 
documentation.

Analysis functions always return :class:`Result <pyprobe.Result>` objects, which allows direct
integration with other PyProBE functionality such as plotting, or other methods.

Analysis functions have been designed to be simple to read and implement. See the 
:ref:`Writing a method <writing_a_method>` section of the 
:ref:`Developer Guide <developer_guide>`. 

.. footbibliography::