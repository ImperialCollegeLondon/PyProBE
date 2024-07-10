.. _plotting:

Plotting
========

PyProBE has a plotting module that can display any :class:`~pyprobe.rawdata.RawData`
or :class:`~pyprobe.result.Result` object. The plotting module is based on
`plotly <https://plot.ly/python/>`_. 

You first create a plot instance:

.. code-block:: python

    plot = pyprobe.Plot()

Then you can add data to and display the plot:

.. code-block:: python

    result = cell.procedure['Procedure Name'].experiment('Experiment Name').cycle(4).charge(2)
    plot.add_line(result, 'Time [s]', 'Voltage [V]')
    plot.show()

:func:`~pyprobe.plot.Plot.add_line` adds a line to the plot. The first argument is the 
:class:`~pyprobe.rawdata.RawData` or :class:`~pyprobe.result.Result` object data to be 
plotted, the second and third arguments are quantities to plot in x and y.

For other plot types see the :class:`~pyprobe.plot.Plot` class documentation.

.. footbibliography::