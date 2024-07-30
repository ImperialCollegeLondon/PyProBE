.. _writing_an_experiment_record:

Writing an Experiment record
============================
An experiment record file is helpful to keep track of information relating to multiple
cells when running experiments. It is used by PyProBE to automatically add to the 
:attr:`~pyprobe.cell.Cell.info` dictionary when a cell list is created.

The experiment record is an excel file that is completely customizable to the user's
needs. The only required column is :code:`Name`, which is used to distinguish cells in
PyProBE's built-in :class:`~pyprobe.plot.Plot` class.

The following is an example of an experiment record file. It includes the key column
:code:`Name`, as well as additional details about the experimental setup.

.. image:: images/Example_experiment_record.png

.. footbibliography::