.. _input_data_guidance:
Input Data Guidance
===================

Supported cyclers and formats
-----------------------------
PyProBE is able to import data from the following cyclers:

* Neware: :code:`'neware'`
  
  - .csv
  - .xlsx

* BioLogic: :code:`'biologic'` or for Modulo Bat files :code:`'biologic_MB'`

  - .mpt
  - .txt

* Arbin: :code:`'arbin'`
  
  - .csv
  - .xlsx

* Maccor: :code:`'maccor'`

  - .csv

* Basytec: :code:`'basytec'`

  - .txt


PyProBE data columns
--------------------
Once converted into the standard PyProBE format, the data columns stored in 
:attr:`RawData.base_dataframe <pyprobe.rawdata.RawData.base_dataframe>` are as follows:

- 'Date' (`polars.datatypes.Datetime <https://docs.pola.rs/py-polars/html/reference/api/polars.datatypes.Datetime.html#polars.datatypes.Datetime>`_): the timestamp
   Date and time the measurement was taken
   
- 'Time [s]' (`polars.datatypes.Float64 <https://docs.pola.rs/py-polars/html/reference/api/polars.datatypes.Float64.html#polars.datatypes.Float64>`_): elapsed time 
   From the start of the filtered data section
- 'Step' (`polars.datatypes.Int64 <https://docs.pola.rs/py-polars/html/reference/api/polars.datatypes.Int64.html#polars.datatypes.Int64>`_): the unique step number 
   Corresponds to a single instruction in the cycling program. Step numbers repeat when instructions are cycled, i.e. the column might look like [1, 1, 1…, 2, 2, 2…, 1, 1, 1…, 2,2,2…, 3, 3, 3…] if steps 1 and 2 were cycled twice
- 'Cycle' (`polars.datatypes.Int64 <https://docs.pola.rs/py-polars/html/reference/api/polars.datatypes.Int64.html#polars.datatypes.Int64>`_): the cycle number
   Automatically identified when Step decreases
- 'Event' (`polars.datatypes.Int64 <https://docs.pola.rs/py-polars/html/reference/api/polars.datatypes.Int64.html#polars.datatypes.Int64>`_): the event number
   Automatically identified when Step changes
- 'Current [A]' (`polars.datatypes.Float64 <https://docs.pola.rs/py-polars/html/reference/api/polars.datatypes.Float64.html#polars.datatypes.Float64>`_): the current in Amperes
   \
- 'Voltage [V]' (`polars.datatypes.Float64 <https://docs.pola.rs/py-polars/html/reference/api/polars.datatypes.Float64.html#polars.datatypes.Float64>`_): the voltage in Volts
   \
- 'Capacity [Ah]' (`polars.datatypes.Float64 <https://docs.pola.rs/py-polars/html/reference/api/polars.datatypes.Float64.html#polars.datatypes.Float64>`_): the capacity passed
   Taken relative to the start of the filtered section in Ampere-hours. Its value increases when charge
   current is passed and decreases when discharge current is passed.

The table below summarises the data columns in the PyProBE format and the corresponding
column names that are required in data from supported cyclers:

.. raw:: html

   <style>
      .scrollable-table-container {
           position: relative;
           overflow-x: auto;
           white-space: nowrap;
      }
      .scrollable-table {
           border-collapse: collapse;
           width: 100%;
      }
      .scrollable-table th, .scrollable-table td {
           padding: 8px;
           text-align: left;
           border: 1px solid #ddd;
      }
      .scrollable-table th {
           background-color: #f2f2f2;
           position: sticky;
           top: 0;
           z-index: 1;
      }
      .scrollable-table th:first-child, .scrollable-table td:first-child {
           position: sticky;
           left: 0;
           z-index: 2;
           background-color: #f2f2f2;
      }
      .scrollable-table th:nth-child(2), .scrollable-table td:nth-child(2) {
           position: sticky;
           left: 115px; /* Adjust this value based on the width of the first column */
           z-index: 2;
           background-color: #f2f2f2;
      }
   </style>

   <div class="scrollable-table-container">

.. table::
   :widths: 20 20 20 20 20 20 20
   :class: scrollable-table
   
   +----------------------+-----------+------------------------+-----------------------------+-----------------------------+-----------------------------+-----------------------------+
   | PyProBE              | Required? | Neware                 | BioLogic                    | Arbin                       | Maccor                      | Basytec                     |
   +======================+===========+========================+=============================+=============================+=============================+=============================+
   | ``Date``             | No        | ``Date``               | ``Acquisition started on``  | ``Date Time``               | ``DPT Time``                | ``~Start of Test``          |
   |                      |           |                        | in header                   |                             |                             | in header                   |
   +----------------------+-----------+------------------------+-----------------------------+-----------------------------+-----------------------------+-----------------------------+
   | ``Time [s]``         | Yes       | *Auto from Date*       | ``time/*``                  | ``Test Time (*)``           | ``Test Time (sec)``         | ``~Time[*]``                |
   +----------------------+-----------+------------------------+-----------------------------+-----------------------------+-----------------------------+-----------------------------+
   | ``Step``             | Yes       | ``Step Index``         | ``Ns``                      | ``Step Index``              | ``Step``                    | ``Line``                    |
   +----------------------+-----------+------------------------+-----------------------------+-----------------------------+-----------------------------+-----------------------------+
   | ``Cycle``            | Yes       | *Auto from Step*       | *Auto from Step*            | *Auto from Step*            | *Auto from Step*            | *Auto from Step*            |
   |                      |           |                        |                             |                             |                             |                             |
   +----------------------+-----------+------------------------+-----------------------------+-----------------------------+-----------------------------+-----------------------------+
   | ``Event``            | Yes       | *Auto from Step*       | *Auto from Step*            | *Auto from Step*            | *Auto from Step*            | *Auto from Step*            |
   |                      |           |                        |                             |                             |                             |                             |
   +----------------------+-----------+------------------------+-----------------------------+-----------------------------+-----------------------------+-----------------------------+
   | ``Current [A]``      | Yes       | ``Current(*)``         | ``I/*``                     | ``Current (*)``             | ``Current``                 | ``I[*]``                    |
   +----------------------+-----------+------------------------+-----------------------------+-----------------------------+-----------------------------+-----------------------------+
   | ``Voltage [V]``      | Yes       | ``Voltage(*)``         | ``Ecell/*``                 | ``Voltage (*)``             | ``Voltage``                 | ``U[*]``                    |
   +----------------------+-----------+------------------------+-----------------------------+-----------------------------+-----------------------------+-----------------------------+
   | ``Capacity [Ah]``    | Yes       | ``Chg. Cap.(*)``,      | ``Q charge/*``,             | ``Charge Capacity (*)``,    | ``Capacity``                | ``Ah[*]``                   |
   |                      |           | ``DChg. Cap.(*)``      | ``Q discharge/*``           | ``Discharge Capacity (*)``  |                             |                             |
   +----------------------+-----------+------------------------+-----------------------------+-----------------------------+-----------------------------+-----------------------------+
   | ``Temperature [C]``  | No        | ``T1(*)``              | ``Temperature/*``           | ``Aux_Temperature_1 (*)``   | ``Temp 1``                  | ``T1[*]``                   |
   +----------------------+-----------+------------------------+-----------------------------+-----------------------------+-----------------------------+-----------------------------+

.. raw:: html

   </div>

Where no units are provided (as is the case with Maccor), the PyProBE default units are
assumed.

The columns marked with *Auto from ...* are automatically generated by the PyProBE 
data import process. This process includes automatic unit conversion to the PyProBE
base units using the :class:`~pyprobe.units.Units` class.







.. footbibliography::