DATABASE = {
        'drivename': 'postgresql',
        'host': 'localhost',
        'port': '5432',
        'username': 'postgresql',
        'password': 'postgresql',
        'database': 'aes_classifier'
        }

try:
    from local_settings import *
except ImportError as e:
    pass
