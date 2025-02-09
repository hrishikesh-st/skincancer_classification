import logging
import os

def get_logger(log_dir, log_file="train.log"):
    """
    Logger for file and console.

    :param log_dir: directory to save log file
    :type log_dir: str
    :return: logger
    :rtype: logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logging.basicConfig(
        filename=log_path,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    logger = logging.getLogger()
    
    # log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    return logger
