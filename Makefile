.PHONY: test

.venv:
	python3 -m venv .venv

install_dev: .venv
	.venv/bin/pip install -e ".[test]"

test: .venv
	.venv/bin/pytest -p no:cacheprovider tests/

table:
	.venv/bin/python misc/generate_table.py

table_timm:
	.venv/bin/python misc/generate_table_timm.py

fixup:
	.venv/bin/ruff check --fix
	.venv/bin/ruff format

all: fixup test

