import base64
import json
import uuid

import config
import cv2
import numpy as np
import redis
from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def index():
    return """
        <html>
            <form action="detect" method="POST" enctype="multipart/form-data">
                Image File: <input type="file" name="image"/><br>
                Object Type: <input type="text" name="object_type"/><br>
                <input type="submit" value="Detect Image" name="submit"/>
            </form>
        </html>
    """


@app.route('/detect', methods=['POST'])
def detect():
    f = request.files['image']
    object_type = request.form['object_type'] \
        if request.form['object_type'] else None

    image = cv2.imdecode(np.frombuffer(f.read(), np.uint8), -1)
    height, width, channels = image.shape

    _id = str(uuid.uuid4())

    # put onto the object detection redis queue
    cache = redis.Redis(host=config.REDIS_HOST)
    cache.rpush(config.REDIS_DETECT_QUEUE, json.dumps({
        'id': _id,
        'image': base64.b64encode(image).decode('utf-8'),
        'height': height,
        'width': width,
        'object_type': object_type,
    }))

    # poll for results
    while True:
        bbs = cache.get(_id)

        if bbs:
            cache.delete(_id)
            break

    return json.loads(bbs)


def main():
    app.run(host='0.0.0.0', port=8080)


if __name__ == '__main__':
    main()
