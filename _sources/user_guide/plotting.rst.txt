.. _plotting:

Plotting
========

PyProBE includes plotting methods that integrate directly with popular Python visualisation
tools. Using a backend powered by `Pandas <https://pandas.pydata.org/>`_ and `matplotlib <https://matplotlib.org/>`_, you can call the 
:func:`~pyprobe.result.Result.plot` method on any :class:`~pyprobe.result.Result` object.

For more interactive plotting, you can use install the optional dependency `hvPlot <https://hvplot.holoviz.org/>`_:

.. code-block:: bash

    pip install 'PyProBE-Data[hvplot]'

This enables the :func:`~pyprobe.result.Result.hvplot` method which creates interactive plots for
visual inspection.

The :func:`~pyprobe.result.Result.plot` and :func:`~pyprobe.result.Result.hvplot` 
interfaces are very similar. For example, creation of a simple plot might look like:

.. code-block:: python

    result = cell.procedure['Procedure Name'].experiment('Experiment Name').cycle(1)
    
    # for matplotlib
    result.plot(x='Time [s]', y='Voltage [V]')

    # for hvplot
    result.hvplot(x='Time [s]', y='Voltage [V]')


PyProBE also includes a wrapper for the `Seaborn <https://seaborn.pydata.org/index.html>`_ 
package. This allows you to pass any :class:`~pyprobe.result.Result` object to the `data`
argument of any seaborn method:

.. code-block:: python

    from pyprobe.plot import seaborn as sns

    sns.scatterplot(result, x='Time [s]', y='Voltage [V]')


Seaborn must be installed as an optional dependency:

.. code-block:: bash

    pip install 'PyProBE-Data[seaborn]'

All of these methods are light wrappers, meaning you can refer to the original package 
documentation for details on methods to customise your plots further. To get started with
plotting view the :doc:`example <../examples/plotting>`.


.. footbibliography::