Accessing Data
==============

.. _filtering:

Filtering
---------

PyProBE is designed to have a simple interface for filtering data. The filtering methods
use chained notation and natural language to be approachable for users who are less 
familiar with Python. Procedure and experiment names are specified as strings by the
user, either :ref:`when the data is imported <adding_data_to_cell>` or in the 
:code:`README.yaml` :ref:`file <writing_a_readme_file>`.

It is possible to filter data by a number of methods:

* First by **procedure**:
   
   .. code-block:: python

      cell.procedure['Procedure Name']

* Then by **experiment**:

   .. code-block:: python

      cell.procedure['Procedure Name'].experiment('Experiment Name')

* Then by the **numerical filter methods** in the :mod:`pyprobe.filters` module:

   .. code-block:: python

      # Filter by cycle
      cell.procedure['Procedure Name'].experiment('Experiment Name').cycle(1)

      # Filter by step
      cell.procedure['Procedure Name'].experiment('Experiment Name').step(3)

      # Filter by cycle then step
      cell.procedure['Procedure Name'].experiment('Experiment Name').cycle(1).step(1)

      # Filter by step type
      cell.procedure['Procedure Name'].experiment('Experiment Name').charge(2)
      cell.procedure['Procedure Name'].experiment('Experiment Name').discharge(0)
      cell.procedure['Procedure Name'].experiment('Experiment Name').rest(1)
      cell.procedure['Procedure Name'].experiment('Experiment Name').chargeordischarge(1)

      # Filter by cycle then step type
      cell.procedure['Procedure Name'].experiment('Experiment Name').cycle(4).charge(2)
      cell.procedure['Procedure Name'].experiment('Experiment Name').cycle(2).discharge(0)
   
   Indices are zero-based, so the first cycle is 0, the first step is 0, etc. The 
   index count is reset after applying every filter, i.e. the first discharge of any 
   cycle is accessed with :code:`discharge(0)`.

RawData objects
---------------
Any filter applied to a cell returns a :class:`~pyprobe.rawdata.RawData` object. This is
a special type of :class:`~pyprobe.result.Result` object that is designed to hold cell
experimental data processed by PyProBE. It therefore has all the attributes of the
:class:`~pyprobe.result.Result` class. This includes:

* :attr:`~pyprobe.result.Result.data` attribute
   a `polars Dataframe <https://docs.pola.rs/py-polars/html/reference/dataframe/index.html>`_
   containing the filtered data.
* :attr:`~pyprobe.result.Result.info` attribute
   the cell's `info` dictionary.

To access the data, you can access the full polars Dataframe:

.. code-block:: python

   dataframe = cell.procedure['Procedure Name'].experiment('Experiment Name').cycle(1).step(1).data

Or you can access individual columns as 1D numpy arrays by calling the 
:func:`~pyprobe.result.Result.get` method:

.. code-block:: python

   voltage, current = cell.procedure['Procedure Name'].experiment('Experiment Name').cycle(1).step(1).get("Voltage [V]", "Current [A]")

Accessing columns directly with this method is useful for converting data to unit 
variants:

.. code-block:: python

   current_mA = cell.procedure['Procedure Name'].experiment('Experiment Name').get("Current [mA]")

To retrieve more than one column, simply pass multiple column names to 
:func:`~pyprobe.result.Result.get` or use :func:`~pyprobe.result.Result.array` to
return an N-dimensional array of the selected columns.

.. footbibliography::