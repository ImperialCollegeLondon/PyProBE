# Contributing to PyProBE

Contributions to PyProBE are welcome. 

If you have a suggestion, please open an issue describing in detail the change you would like to be made. Similarly, if you have found a bug or a mistake, please open an issue describing this in detail.

If you would like to contribute code, please:

1. Fork the repository

2. Install the development requirements:

```bash
$ source .venv/bin/activate
$ pip install -r requirements-dev.txt
```

3. Install the git pre-commit hooks:

```bash
$ pre-commit install
```

4. Open an issue to detail the change/addition you wish to make, unless one already exists

5. Create a feature branch and make your changes

6. Follow [Google's docstring style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings) and ensure that the [documentation builds](#viewing-the-api-documentation) successfully:

```bash
$ cd docs
$ make html
```

7. Ensure that all tests pass:

```bash
$ pytest
```

8. Open a pull request. In the pull request description, please describe in detail the changes your feature branch introduces, and reference the associated issue.

## Packages and dependencies
You are welcome to suggest and add new dependencies, if required for your functionality. There are two limits to this:
1. DataFrame operations should only be done using polars expressions. Data should be kept by default in polars LazyFrame format and only converted to DataFrame if needed for a particular operation.
2. Input variables to methods should be 1-Dimensional numpy arrays.

## Viewing the API documentation

API documentation is built in html format, and stored locally in docs/build/html/. This can be viewed in your browser at docs/build/html/index.html.

The documentation is also continuously deployed via GitHub Actions, and can be viewed [here](https://congenial-adventure-mz797n5.pages.github.io).