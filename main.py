# -*- encoding: utf-8 -*-

import logging
from logging.config import dictConfig

<<<<<<< HEAD
from module5.ann import ANN
from module5.mnist import mnist_basics
from module5.myth_testing import debug
=======
from module5.ann import ANN, rectify, softmax
>>>>>>> 99615fdb895a8152e009837aa66e49ff385ed0be

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
    layer_structure = [784, 620, 10]
    activation_functions = [rectify, rectify, softmax]

    # Create a network using the default parameters
    a = ANN(layer_structure, activation_functions)
    a.load_input_data()
    a.train(epochs=5)
    feature_sets, labels = mnist_basics.load_all_flat_cases(type="testing")
    print(a.blind_test(feature_sets[:10]))
    # a.predict(some_784_element_long_flat_image_vector)
