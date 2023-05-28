.PHONY: test

.venv:
	python3 -m venv .venv

install_dev: .venv
	.venv/bin/pip install -e .[test]
	.venv/bin/pre-commit install

test: .venv
	.venv/bin/pytest -p no:cacheprovider tests/

table:
	.venv/bin/python misc/generate_table.py

table_timm:
	.venv/bin/python misc/generate_table_timm.py

black: .venv
	.venv/bin/black ./segmentation_models_pytorch --config=pyproject.toml --check

flake8: .venv
	.venv/bin/flake8 ./segmentation_models_pytorch --config=.flake8

all: black flake8 test
