# src/makeup_config.py

import json
from collections import namedtuple
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# Define a namedtuple for makeup type configurations
MakeupTypeConfig = namedtuple('MakeupTypeConfig', [
    'name',
    'facemesh_regions',
    'default_color',
    'default_intensity'
])

def load_makeup_configs(config_path='configs/makeup_types.json') -> list:
    """
    Loads makeup configurations from a JSON file.
    
    :param config_path: Path to the JSON configuration file.
    :return: List of MakeupTypeConfig instances.
    """
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    makeup_configs = []
    for item in data:
        makeup_configs.append(
            MakeupTypeConfig(
                name=item['name'],
                facemesh_regions=item['facemesh_regions'],
                default_color=tuple(item['default_color']),
                default_intensity=item['default_intensity']
            )
        )
    
    logging.info(f"Loaded {len(makeup_configs)} makeup configurations.")
    return makeup_configs

# Load configurations at module import
MAKEUP_TYPES_CONFIG = load_makeup_configs()
