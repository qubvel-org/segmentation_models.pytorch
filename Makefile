.PHONY: test  # Declare the 'test' target as phony to avoid conflicts with files named 'test'

# Variables to store the paths of the python, pip, pytest, and ruff executables
PYTHON := $(shell which python)
PIP := $(shell which pip)
PYTEST := $(shell which pytest)
RUFF := $(shell which ruff)

# Target to create a Python virtual environment
.venv:
	$(PYTHON) -m venv $(shell dirname $(PYTHON))

# Target to install development dependencies in the virtual environment
install_dev: .venv
	$(PIP) install -e ".[test]"

# Target to run tests with pytest, using 2 parallel processes and only non-marked tests
test: .venv
	$(PYTEST) -v -rsx -n 2 tests/ --non-marked-only

# Target to run all tests with pytest, including slow tests, using 2 parallel processes
test_all: .venv
	RUN_SLOW=1 $(PYTEST) -v -rsx -n 2 tests/

# Target to generate a table by running a Python script
table:
	$(PYTHON) misc/generate_table.py

# Target to generate a table for timm by running a Python script
table_timm:
	$(PYTHON) misc/generate_table_timm.py

# Target to fix and format code using ruff
fixup:
	$(RUFF) check --fix
	$(RUFF) format

# Target to run code formatting and tests
all: fixup test
