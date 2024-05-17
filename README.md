# PyProBE
PyProBE is a Python library for the analysis of battery cycling data.

## Using PyProBE
See the [documentation](https://congenial-adventure-mz797n5.pages.github.io) for a detailed getting started guide.

You can find example notebooks in the [examples folder](examples/).

PyProBE should be imported as a package into an external post-processing script or jupyter notebook.

```python
from pyprobe.cell import Cell
```

You can make a PyProBE Cell object with only a dictionary of metadata about the cell being cycled:
```python
info = dict({'metadata item': value})
cell = Cell(info)
```

To add data to your cell object, it must first be converted into the standard format used by PyProBE. See the list of currently supported cyclers and file formats in the documentation.

```python
cell.process_cycler_file(cycler = 'supported_cycler',
                         folder_path = 'root_directory/subfolder',
                         file_name = 'cell_data.anyfomat')
```

To add the newly created ```.parquet``` file to the cell object, you must have a ```README.yaml``` file saved alongside your raw data, following the guidance in the documentation.

```python
cell.add_procedure(procedure_name = 'procedure name', 
                   input_path = 'root_directory/subfolder', 
                   file_name = 'cell_data.parquet')
```

Batch processing can also be done. This requires an ```Experiment_Record.xlsx``` in the ```root directory```, according to the guidelines in the documentation. For more information see the documentation and this example.

## Installing PyProBE
<details>
  <summary>Linux/macOS</summary>

  1. Clone the repository and enter the directory:
  ```bash
  $ git clone https://github.com/ImperialCollegeLondon/PyProBE.git
  ```

  2. Create and activate a virtual environment:
  
  venv (in your working directory):
```bash
$ python -m venv .venv
$ source .venv/bin/activate
```
conda (in any directory):
```bash
$ conda create -n pyprobe python=3.12
$ conda activate pyprobe
```


  3. Install PyProBE as a package into your virtual environment:
  ```bash
  $ cd PyProBE
  $ pip install .
  ```
</details>

<details>
  <summary>Windows</summary>

  1. Clone the repository and enter the directory:
  ```bat
  > git clone https://github.com/ImperialCollegeLondon/PyProBE.git
  ```

  2. Create and activate a virtual environment:
  
  venv (in your working directory):
```bat
> python -m venv .venv
> source .venv/bin/activate
```
conda (in any directory):
```bash
> conda create -n pyprobe python=3.12
> conda activate pyprobe
```

  3. Install PyProBE as a package into your virtual environment:
  ```bat
  > cd PyProBE
  > pip install .
  ```
</details>

## Citing PyProBE

TBC


## Contributing to PyProBE

Contributions to PyProBE are welcome. Please see the [contributing guidelines](CONTRIBUTING.md).


## License

PyProBE is fully open source. For more information about its license, see [LICENSE](LICENSE.md).


## Contributors
<!-- readme: contributors -start -->
<table>
<tr>
    <td align="center">
        <a href="https://github.com/tomjholland">
            <img src="https://avatars.githubusercontent.com/u/137503955?v=4" width="100;" alt="tomjholland"/>
            <br />
            <sub><b>Tom Holland</b></sub>
        </a>
    </td></tr>
</table>
<!-- readme: contributors -end -->
