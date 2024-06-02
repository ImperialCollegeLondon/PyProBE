Using Data
==========

Filtering
---------

PyProBE is designed to have a simple interface for filtering data. The filtering methods
use chained notation and natural language to be approachable for users who are less 
familiar with Python. Procedure and experiment names are specified as strings by the
user, either :ref:`when the data is imported <adding_data_to_cell>` or in the 
:code:`README.yaml` :ref:`file <writing_a_readme_file>`.

It is possible to filter data by a number of methods:

* First **procedure**:
   
   .. code-block:: python

      cell.procedure('Procedure Name')

* Then by **experiment**:

   .. code-block:: python

      cell.procedure('Procedure Name').experiment('Experiment Name')

* Then by the **numerical filter methods** in the :class:`pyprobe.filter.Filter` class:

   .. code-block:: python

      # Filter by cycle
      cell.procedure('Procedure Name').experiment('Experiment Name').cycle(1)

      # Filter by step
      cell.procedure('Procedure Name').experiment('Experiment Name').step(3)

      # Filter by cycle then step
      cell.procedure('Procedure Name').experiment('Experiment Name').cycle(1).step(1)

      # Filter by step type
      cell.procedure('Procedure Name').experiment('Experiment Name').charge(2)
      cell.procedure('Procedure Name').experiment('Experiment Name').discharge(0)
      cell.procedure('Procedure Name').experiment('Experiment Name').rest(1)
      cell.procedure('Procedure Name').experiment('Experiment Name').chargeordischarge(1)

      # Filter by cycle then step type
      cell.procedure('Procedure Name').experiment('Experiment Name').cycle(4).charge(2)
      cell.procedure('Procedure Name').experiment('Experiment Name').cycle(2).discharge(0)
   
   Indices are zero-based, so the first cycle is 0, the first step is 0, etc. The 
   index count is reset after applying every filter, i.e. the first discharge of any 
   cycle is accessed with :code:`discharge(0)`.

