# Contributing to PyProBE

Contributions to PyProBE are welcome. 

If you have a suggestion, please open an issue describing in detail the change you would like to be made. Similarly, if you have found a bug or a mistake, please open an issue describing this in detail.

If you would like to contribute code, please:

1. Install PyProBE with [developer settings](https://congenial-adventure-mz797n5.pages.github.io/installation.html)

2. Open an issue to detail the change/addition you wish to make, unless one already exists

3. Create a feature branch and make your changes

4. Follow [Google's docstring style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings) and ensure that the [documentation builds](#viewing-the-api-documentation) successfully:

```bash
$ cd docs
$ make html
```

5. Ensure that all tests pass:

```bash
$ pytest
```

6. Open a pull request. In the pull request description, please describe in detail the changes your feature branch introduces, and reference the associated issue.

## PyProBE structure
Additions to the code should be made in accordance with the structure of PyProBE, to 
maximise compatibility and ensure it is a maintainable package. Guidance for writing
code for PyProBE includes:
1. DataFrame operations should only be done using polars expressions. Data should be kept by default in polars LazyFrame format and only converted to DataFrame if needed for a particular operation.
2. Method classes should be written in the format described in the [documentation](
    https://congenial-adventure-mz797n5.pages.github.io/creating_a_new_method.html
).

## Viewing the API documentation

API documentation is built in html format, and stored locally in docs/build/html/. This can be viewed in your browser at docs/build/html/index.html.

The documentation is also continuously deployed via GitHub Actions, and can be viewed [here](https://congenial-adventure-mz797n5.pages.github.io).