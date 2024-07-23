.. _creating_a_new_method:

Creating a New Method
=====================
In PyProBE :class:`analysis <pyprobe.methods.basemethod.BaseMethod>` are classes that 
perform further analysis of the data. They can be peformed on any 
:class:`~pyprobe.rawdata.RawData` or :class:`~pyprobe.result.Result` object.

This document describes the standard format to be used for all PyProBE methods. 
Constructing your method in this way ensures compatibility with the rest of the 
PyProBE package, while keeping your code clean and easy to read.

Start by creating your class, and an __init__ method. This is where all of the 
variable assignement is performed:
1. Define the input variables to the method using the 
   :meth:`~pyprobe.methods.basemethod.BaseMethod.variable` method.
2. Perform the calculations using the input variables and any other functions defined
   inside this method class.
3. Assign the results to the class attributes, which allows them to be read as 
   :class:`~pyprobe.result.Result` objects.

Then you can add any additional functions to perform your calculations. Keep these
simple, defining all inputs as numpy arrays with minimal dimensionality. For instance,
in the example below, current and time are both 1D numpy arrays. The code could have
performed the same calculations using a single 2D array with a column for each variable,
however this would have made the code less readable.

.. code-block:: python

    from pyprobe.methods.basemethod import BaseMethod

    class CoulombCounter(BaseMethod):
      """"A method for calculating the charge passed.""""

        def __init__(self, input_data: Result):
            """Initialise the coulomb counter method.
            
            Args:
                data: The input data to the method.
            """
            # define input variables to the method
            self.current = self.variable("Current [A]")
            self.time = self.variable("Time [s]")

            # perform the calculations
            self.charge = self.coulomb_count(self.current, self.time)

            # assign the results, by calling the make_result method on a dictionary
            # of the results of your method
            self.result = self.make_result({"Charge [C]": self.charge})
        
        def coulomb_count(self, current, time):
            """Calculate the charge passed.
            
            Args:
                current: The current data.
                time: The time data.
            
            Returns:
                The charge passed.
            """
            return np.trapz(current, time)

The result of the method above can be accessed as a result object by calling:

.. code-block:: python

    result = CoulombCounter(data).result

Which can be passed to any other Method or used in the :class:`~pyprobe.plot.Plot` 
class.

.. footbibliography::