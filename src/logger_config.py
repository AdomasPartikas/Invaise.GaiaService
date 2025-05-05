import logging
import sys
import os
from datetime import datetime

import config

def setup_logging():
    """
    Set up logging configuration for all modules.
    """
    # Ensure data directory exists
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Clear any existing handlers to avoid duplicate logs
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    
    # Add handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Initial log messages to verify logging is working
    logging.info(f"Logging initialized at {datetime.now().isoformat()}")
    logging.info(f"Log level: {config.LOG_LEVEL}")
    logging.info(f"Log file: {config.LOG_FILE}")
    
    return root_logger 