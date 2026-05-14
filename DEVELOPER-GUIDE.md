# Developer Guide


## Development Policies

Development is done on GitHub at [https://github.com/GabrieleDiCorato/abides-ng](https://github.com/GabrieleDiCorato/abides-ng).

### Git Processes

- Development is done on the `dev` branch. At certain intervals this code will be merged
  into the `main` branch.
- Any commit that affects code should be developed on a separate branch
  and merged with a pull request.
- Pull requests should have at least one reviewer. This can be assigned on GitHub.


### Dev Branch Merge Requirements

- Code must be documented:
    - This includes documentation of the actual code and documentation of any functions
      and classes.
    - New classes should be accompanied with a new Sphinx documentation page.

- The full test suite (`pytest`) must pass with no regressions.

- Code must be formatted according to the [black](https://github.com/psf/black)
  formatter tool style guidelines.

### Prod Branch Merge Requirements

- Commits to the `prod` branch should only be merges from the `dev` branch.
- An exception to this is hotfixes.
- Any breaking changes (of API or outputs) compared to the previous commit should be
  clearly documented and communicated to the team.


## Code Style

- Code style should follow [PEP8](https://www.python.org/dev/peps/pep-0008/).
- Code must also be formatted according to the [black](https://github.com/psf/black)
formatting tool.
- Type annotations should be provided for function signatures and class attribute declarations.

- [isort](https://pypi.org/project/isort/) is suggested for organising imports.

- [pre-commit](https://pre-commit.com/) can be used for automatically applying these changes
  when committing.

## Setting up pre-commit hooks

Git hooks are tasks that run after a commit is created but before it is confirmed and entered
into the git history.

These tasks can potentially stop a commit from being confirmed.

The [pre-commit](https://pre-commit.com/) tool is used to manage these hooks.

The configuration can be found in the `.pre-commit-config.yaml` file in the repository root.

Currently the following hooks are enabled:

- `black`: auto-formats Python code on every commit.
- `isort`: sorts and groups imports (using the `black` profile).
- `ruff`: lints Python code and applies safe auto-fixes.
- `mypy`: runs static type checking.
- `pytest-check`: runs the full unit test suite and blocks the commit on failure.
- `nb-clean`: strips outputs and transient metadata from `.ipynb` files before
  committing, keeping notebook diffs readable and repositories small. Do not
  commit notebook outputs manually — let this hook handle it.
- Standard file checks: trailing whitespace, end-of-file newline, YAML validity,
  large file detection, and merge-conflict markers.


### Commands

To install pre-commit run:
```
$ uv sync --dev
# or with pip
$ pip install -e .[dev]
```

To enable pre-commit hooks run:
```
$ pre-commit install
```

To disable pre-commit hooks run:
```
$ pre-commit uninstall
```

To test the pre-commit hooks without creating a git commit run:
```
$ pre-commit run --all-files
```

## Testing


### Unit Tests

ABIDES uses the pytest framework for unit tests. All tests are contained within
`tests/` directories within the respective sub-project directories.

### Regression Testing

There is no automated regression framework. Use the smoke test script
`abides-core/scripts/evaluate_all_agents.py` to verify end-to-end behaviour
across the main agent types and result profiles:

```bash
uv run python abides-core/scripts/evaluate_all_agents.py
```



## Documentation

- Code should be documented when written.
- Classes and functions should have [Google Style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
  doc strings.
- Non-code documentation should be placed in the `docs/` directory in RST format.

### Writing Documentation Strings

```python
class ExampleClass:
    """A short sentence describing the class should be on the first row.

    Triple quotes are used for documentation comments.

    Each class should have a description describing the class and it's functionality.

    Class attributes that are relevant for users of the class should be given as follows:

    Attributes:
        price: The order limit price.
        quantity: The amount of shares traded.

    Types are not needed here as they will be taken from type annotations.

    Single backticks can be used to highlight code strings. E.g. `ExampleClass`.
    """

    def do_something(self, x: int, y: int) -> bool:
        """Does something with x and y.

        The same style should be used for functions and class methods. Note the type
        annotations.

        Here we document the function/method arguments and return value if needed:

        In class methods we do not need to document the `self` parameter.

        Arguments:
            x: The first number.
            y: The second number.

        Returns:
            True if x > y else False

        """

        return x > y
```

### Building Documentation Pages

To build the documentation pages run the following:

```bash
sphinx-build -M html docs/ docs/_build
```

To view the pages in a web browser on your local machine you can start a Python web server:

``` bash
cd docs/_build/html && python3 -m http.server 8080
```

Then navigate to [http://localhost:8080]() in your web browser.


## Useful Links

- PEP8: Official Python Style Guide:
    [https://www.python.org/dev/peps/pep-0008/]()

- Type annotations cheat sheet:
    [https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html]()

- 'Google Style' Python documentation guide:
    [https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html]()

- Black code formatter:
    [https://black.readthedocs.io/en/stable/]

- Sphinx Doc basics guide:
    [https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html]()
