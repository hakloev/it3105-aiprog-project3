# -*- encoding: utf-8 -*-

import logging
from logging.config import dictConfig

from module5.ann import ANN
from module5.myth_testing import debug

LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'default',
            'filename': 'debug.log',
            'maxBytes': 1024 * 1024 * 10,
            'backupCount': 1
        }
    },
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s %(name)s.%(funcName)s:%(lineno)d %(message)s'
        }
    },
    'loggers': {
        '': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}

if __name__ == "__main__":

    # Set up logging
    dictConfig(LOG_CONFIG)
    log = logging.getLogger(__name__)

    # Network structure
    # Structure: [input_layer, hidden_layer, hidden_layer ... , output_layer]
    # Example: [784, 620, 100, 10]
    layer_structure = [784, 784, 620, 10]

    # Create a network using the default parameters
    #debug()
    a = ANN(layer_structure)
    a.train(runs=50)
    # a.predict(some_784_element_long_flat_image_vector)
