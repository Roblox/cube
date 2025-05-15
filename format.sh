#!/bin/bash
set -e

python -m ruff format .
python -m ruff check . --fix
# Sort imports
python -m ruff check . --select I --fix