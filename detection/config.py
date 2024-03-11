import os

REDIS_HOST = os.environ.get('REDIS_HOST', 'redis')
REDIS_DETECT_QUEUE = 'detect'
