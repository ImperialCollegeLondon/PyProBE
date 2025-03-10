# Contributing to PyProBE

Contributions to PyProBE are welcome. 

If you have a suggestion, please open an issue describing in detail the change you would like to be made. Similarly, if you have found a bug or a mistake, please open an issue describing this in detail.

If you would like to contribute code, please:

1. Install PyProBE with [developer settings](https://pyprobe.readthedocs.io/en/latest/developer_guide/developer_installation.html)

2. Open an issue to detail the change/addition you wish to make, unless one already exists

3. Create a feature branch and make your changes. PyProBE uses the [angular commit style](https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#commits), please ensure your commits follow this syntax before pushing your changes

4. Follow [Google's docstring style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings) and ensure that the documentation builds successfully:

```bash
$ cd docs
$ make html
```

5. Ensure that all tests pass (this is best done with uv if you have installed with this tool):

```bash
$ uv run pytest
```

6. Ensure that the examples run to completion:
```bash
$ uv run pytest --nbmake docs/source/examples/*.ipynb
```

7. Open a pull request. In the pull request description, please describe in detail the changes your feature branch introduces, and reference the associated issue.

## PyProBE structure
Additions to the code should be made in accordance with the structure of PyProBE, to 
maximise compatibility and ensure it is a maintainable package. Guidance for writing
code for PyProBE includes:
1. DataFrame operations should only be done using polars expressions. Data should be kept by default in polars LazyFrame format and only converted to DataFrame if needed for a particular operation.
2. Analysis classes should be written in the format described in the [documentation](https://pyprobe.readthedocs.io/en/latest/developer_guide/contributing_to_the_analysis_module.html).

## Linting and Style Guidelines
PyProBE uses [Ruff](https://docs.astral.sh/ruff/) to check and format code against Python standards and good practise.
It is able to automatically restyle your code and can make many automatic fixes for you. It 
runs as a pre-commit hook, meaning it should pass before you commit to PyProBE. To reduce 
the burden of making a large amount of fixes, be sure to make regular commits. You can also run:
```bash
uvx ruff check --fix
```
from the command line at any point. This will load the latest version of Ruff and run its checks, 
making automatic fixes where possible.

PyProBE also uses [mypy](https://mypy.readthedocs.io/en/stable/index.html) to check that
all functions are correctly type hinted. Again, this runs as a pre-commit hook. It is
likely that your code will use types already commonly used in PyProBE, so you may refer
to existing code for how to type hint your functions.

## Viewing the API documentation

API documentation is built in html format, and stored locally in docs/build/html/. This can be viewed in your browser at docs/build/html/index.html.

The documentation is also continuously deployed via GitHub Actions, and can be viewed [here](https://pyprobe.readthedocs.io).
