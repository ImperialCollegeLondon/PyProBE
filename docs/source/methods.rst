Methods
=======

Once filtered, PyProBE methods are there to perform further analysis on your 
experimental data. They are contained within the :mod:`methods <~pyprobe.methods>` 
module, and you can see the currently available methods in this part of the API 
documentation.

Methods always return :class:`Result <pyprobe.Result>` objects, which allows direct
integration with other PyProBE functionality such as plotting, or other methods.

Methods have been designed to be simple to read and implement. See the 
:ref:`Writing a method <writing_a_method>` section of the 
:ref:`Developer Guide <developer_guide>`. 