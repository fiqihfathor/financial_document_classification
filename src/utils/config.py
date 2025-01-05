import yaml    
from loguru import logger
import sys

def load_config(config_file: str='./config/config.yml'):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        raise
    

def setup_logging(config_path: str = './config/logging.yml'):
    """Load logging configuration from YAML file and set up loguru."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    logger.remove()

    console_handler = config['handlers'].get('console', None)
    if console_handler:
        logger.add(sys.stdout,
                   level=console_handler.get('level', 'INFO'),
                   format=console_handler.get('format', '{message}'))

    file_handler = config['handlers'].get('file', None)
    if file_handler:
        logger.add(file_handler.get('filename', 'logs/default.log'),
                   level=file_handler.get('level', 'INFO'),
                   format=file_handler.get('format', '{message}'),
                   rotation=file_handler.get('rotation', '1 day'),
                   retention=file_handler.get('retention', '7 days'),
                   compression=file_handler.get('compression', 'zip'))


    return logger
