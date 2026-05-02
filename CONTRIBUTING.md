# Contributing to PyProBE

Contributions are welcome. Open an issue first to discuss changes, then follow these steps.

## Setup

1. Clone the repository and navigate to it.
2. Install uv if not already installed: https://docs.astral.sh/uv/getting-started/installation/
3. Install dependencies and pre-commit hooks:
```bash
uv sync
uv run pre-commit install
```
4. Verify setup by running tests:
```bash
uv run pytest
```

## Code Standards

- **DataFrame ops:** polars expressions only. Keep data as `LazyFrame`; convert to `DataFrame` only when needed. Numpy only for complex logic in `pyprobe/analysis/`.

- **Docstrings:** Google style.

- **Type hints:** required on all public function signatures. Use 3.10+ syntax:
  - `X | None` not `Optional[X]`
  - `list[int]` not `List[int]`
  - `tuple[str, ...]` not `Tuple[str, ...]`

- **Exceptions:** use specific types (`ValueError`, `TypeError`, `KeyError`, etc.). No bare `except:` or `except Exception:` without re-raise.

- **Comments:** inline only for non-obvious logic. No dashed-line block headers.

- **PEP 8:** line length 88.

- **Logging:** use loguru. Import as `from loguru import logger`. Log at appropriate levels: `debug` for development, `info` for key events, `warning` for recoverable issues, `error` for failures.

## Test Standards

- Tests mirror `pyprobe/` structure in `tests/` directory.
- Write tests with pytest. Cover expected behavior and edge cases for all public functions.
- One logical behaviour per test function. Name tests as `test_<unit>_<scenario>_<expected>` (e.g., `test_cell_filter_empty_result_raises_error`).
- Keep tests independent; use pytest fixtures for shared setup.
- Use `@pytest.mark.parametrize` for multiple inputs instead of duplicating test bodies.
- Use `tmp_path` fixture for temporary files; never `tempfile` directly.
- Simple synthesised test data should be used where possible; real-format sample files in `tests/sample_data/` where required.

## Commit Messages

Use [Angular commit style](https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#commits).

## Before Opening a Pull Request

Pre-commit checks will run automatically. You can invoke them manually with:
```bash
uvx pre-commit run --all-files
```

Build docs:
```bash
cd docs && make html
```

Run tests:
```bash
uv run pytest
uv run pytest --nbmake docs/source/examples/*.ipynb
```

## Pull Request

Describe changes in detail and reference the associated issue.
