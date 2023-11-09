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

precommit: .venv
	.venv/bin/pre-commit run --all-files

all: precommit test
