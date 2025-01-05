SRC_DIR = $(pwd)/src
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
DOCKERFILE = Dockerfile
COMPOSE_FILE = docker-compose.yml
SCRIPTS_DIR = scripts

.PHONY: dataset train server test tracking docs

dataset:
	sh $(SCRIPTS_DIR)/download_data.sh

train:
	@set -x;export PYTHONPATH=$(PYTHONPATH):$(SRC_DIR) && .venv/bin/python $(SCRIPTS_DIR)/train.py

tracking:
	.venv/bin/python -m mlflow ui --port 5000

server:
	export PYTHONPATH=$(PYTHONPATH):$(SRC_DIR) && .venv/bin/python $(SCRIPTS_DIR)/server.py

test:
	.venv/bin/python -m pytest $(SRC_DIR)

docs:
	cd docs && uvx mkdocs serve