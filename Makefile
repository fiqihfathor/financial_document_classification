SRC_DIR = src
VENV_DIR = .venv
DOCKERFILE = Dockerfile
COMPOSE_FILE = docker-compose.yml
SCRIPTS_DIR = scripts

dataset: 
	sh $(SCRIPTS_DIR)/download_data.sh