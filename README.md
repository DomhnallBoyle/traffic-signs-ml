# traffic-signs-ml

Detection and classification of traffic lights using YOLOv3 and Bosch traffic lights dataset

### Requirements:
- [dataset](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset)
- ```pip install -r requirements.txt```

### Running:
- ```classification/pre_process_bosch.py``` - crop traffic lights, save to disk, create dataset csv
- ```classification/inception_v3.py``` - apply transfer learning to traffic lights dataset
- ```detection/server.py``` - Flask API to detect an object using YOLOv3
- ```detection/detect.py``` - YOLOv3 detection
- ```yolo_to_keras.py``` - convert YOLO weights for use in Keras

