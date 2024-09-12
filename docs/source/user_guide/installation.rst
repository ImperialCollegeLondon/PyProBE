Installation
============

To install PyProBE you must be running Python 3.11 or later. It is recommended to use a 
virtual environment to install PyProBE, for example venv or conda.

.. dropdown:: If you are completely new to Python

   **Recommended software**

   The easiest way to get started with data processing in Python is to use 
   `Anaconda <https://docs.anaconda.com/anaconda/install/>`_ for package management and
   `Visual Studio Code <https://code.visualstudio.com/download>`_ for code editing. You 
   will need `Git <https://git-scm.com/>`_ installed to clone the repository. If you 
   are new to Git version control, the `GitHub Desktop <https://github.com/apps/desktop>`_
   is a good place to start.

   In order to follow the installation instructions below, on Windows you can work in
   Anaconda Prompt. On Mac or Linux you can use Terminal directly.

   **Using PyProBE after installation**

   Jupyter Notebooks are a popular format for Python data processing. VSCode has 
   support for writing and running these, which you can open from the dropdown menu:

   .. image:: images/VSCode_open_file.png

   You should then select the Anaconda environment that you will create from the list of 
   available Python environments:

   .. image:: images/VSCode_select_kernel.png


You should have two separate directories: 

* one for the PyProBE repository 
* one for your processing script

The steps to install PyProBE are as follows:

1. Enter a directory in which you wish to install PyProBE:
   
   .. code-block:: bash

      cd /path/to/installation/directory

2. Clone the repository to your local machine. This creates a directory called PyProBE.

   .. code-block:: bash

      git clone https://github.com/ImperialCollegeLondon/PyProBE.git
      cd PyProBE

   It is recommended to used the tagged releases for a stable version of the code. 
   You can list the available tags with the following command:

   .. code-block:: bash

      git tag

   You can then checkout the desired tag with the following command:
   
   .. code-block:: bash

      git checkout <tag name>

   For example:

   .. code-block:: bash

      git checkout v0.1.0


   To update your installation you can run:
   
   .. code-block:: bash

      git fetch --tags
      git tag
      git checkout <tag name>

3. Create and activate a virtual environment.
  
  .. tabs::
      .. tab:: venv

         In your working directory:

         .. code-block:: bash

            python -m venv venv
            source .venv/bin/activate

      .. tab:: conda
            
         In any directory:

         .. code-block:: bash

            conda create -n pyprobe python=3.12
            conda activate pyprobe

3. Install PyProBE's dependencies:
   
   .. code-block:: bash

      cd /path/to/your/directory/PyProBE
      pip install -r requirements.txt

4. Install PyProBE as a package into your virtual environment:
   
   .. code-block:: bash

      pip install .

5. In a second directory you can create a new python script or jupyter notebook to 
   process your data. You can import PyProBE into your script as follows:

   .. code-block:: python

      import pyprobe

6. Before being able to launch the dashboard you will need to initialise streamlit.
   Do this by running the streamlit Hello app from your command line:

   .. code-block:: bash

      streamlit hello

.. footbibliography::