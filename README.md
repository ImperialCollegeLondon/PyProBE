# PyProBE
PyProBE is a Python library for the analysis of battery cycling data.

PyProBE is designed to:
- Be easy to use
- Accelerate battery data exploration
- Encourage data processing under FAIR principles
- Host a library of post-processing methods

## Installing PyProBE
Please follow the [user installation guide](https://congenial-adventure-mz797n5.pages.github.io/installation.html) to install PyProBE.

## Getting Started with PyProBE
PyProBE should be imported as a package into an external post-processing script or jupyter notebook.

```python
import pyprobe
```

PyProBE ```Cell``` objects enable easy access to data throughout the life of a cell undergoing lab experiments on a battery cycler. Data from multiple procedures can be accessed with natural language function calls:
```python
cell.procedure['Reference Test']
cell.procedure['Cycling']
```

Which can be filtered further into the experiments that make up the procedure:
```python
cell.procedure['Reference Test'].experiment('C/10 pOCV')
cell.procedure['Reference Test'].experiment('Discharge Pulses')
```
And/or filtered by cycle, step or step type:
```python
cell.procedure['Reference Test'].step(1)
cell.procedure['Reference Test'].experiment('Discharge Pulses').cycle(3).discharge(0)
```

The PyProBE Dashboard can be launched on a list of ```Cell``` objects to allow rapid data exploration and plotting:
![PyProBE Dashboard](./docs/source/user_guide/images/Dashboard.png)

See the [documentation](https://congenial-adventure-mz797n5.pages.github.io) for a detailed user guide. Start with the following pages to get PyProBE set up with your data:
- [Importing data](https://congenial-adventure-mz797n5.pages.github.io/importing_data.html)
- [Accessing data](https://congenial-adventure-mz797n5.pages.github.io/accessing_data.html)
- [Plotting](https://congenial-adventure-mz797n5.pages.github.io/plotting.html)

You can find example notebooks in the [examples folder](examples/).

## Input data guidelines
PyProBE works with numerous cyclers. For guidance on how to export your data to work with PyProBE see the [Input Data Guidance](https://congenial-adventure-mz797n5.pages.github.io/input_data_guidance.html).

The PyProBE workflow encourages thorough documentation of experimental data. For guidance please see:
- [README file guidelines](https://congenial-adventure-mz797n5.pages.github.io/writing_a_readme_file.html)
- [Experiment record guidelines](https://congenial-adventure-mz797n5.pages.github.io/writing_an_experiment_record.html#)

## Data Analysis Tools
PyProBE provides easy access to tools for further analysis of battery data, including:
- Capacity loss calculation during cycling
- dQ/dV on OCV data
- OCV fitting and Degradation Mode Analysis (DMA)

PyProBE's [Method](https://congenial-adventure-mz797n5.pages.github.io/creating_a_new_method.html) framework allows new data analysis tools to be added quickly and easily.

## Contributing to PyProBE

Contributions to PyProBE are welcome. Please see the [contributing guidelines](CONTRIBUTING.md).

## Citing PyProBE

TBC


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
