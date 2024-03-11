"""Pre-process the bosch traffic lights dataset"""
import argparse
import os
import shutil
import yaml

import cv2
import pandas as pd


def main(args):
    with open(args.yaml_path, 'r') as f:
        contents = yaml.safe_load(f)

    dataset_directory = os.path.dirname(args.yaml_path)
    output_directory = os.path.join(dataset_directory, 'pre_processed')
    output_csv = os.path.join(output_directory, 'dataset.csv')

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    csv_data = []

    for image_content in contents:
        boxes = image_content['boxes']
        if not boxes:
            continue

        image_path = image_content['path']
        image_id = image_path.split('/')[-1].replace('.png', '')
        abs_image_path = os.path.join(dataset_directory, image_path)

        image = cv2.imread(abs_image_path, 1)

        for i, box in enumerate(boxes):
            label = box['label']
            x_min, y_min, x_max, y_max = int(box['x_min']), int(box['y_min']),\
                                         int(box['x_max']), int(box['y_max'])
            width = x_max - x_min
            height = y_max - y_min
            cropped_traffic_light = image[y_min:y_min+height, x_min:x_min+width]

            # cv2.imshow('TL', cropped_traffic_light)
            # cv2.waitKey(0)

            output_image_path = \
                f'{os.path.join(output_directory, image_id)}_{i+1}.png'
            try:
                cv2.imwrite(output_image_path, cropped_traffic_light)
                csv_data.append([output_image_path, label])
            except cv2.error:
                print(f'{output_image_path} failed to write')
                pass

    df = pd.DataFrame(columns=['image_path', 'label'], data=csv_data)
    df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('yaml_path')

    main(parser.parse_args())
