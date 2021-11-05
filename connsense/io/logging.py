import os
import logging

def get_logger(for_step, at_level=None):
    """Centralized logging for ConnSense Apps."""
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(f"ConnSense {for_step.upper()}")
    logger.setLevel(at_level if at_level
                    else os.environ.get("LOGLEVEL", "INFO"))
    return logger
