import argparse

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

HEIGHT, WIDTH, CHANNELS = (75, 200, 3)
# we only predict 4 classes: Red, Yellow, Green and off
LABELS = {
    'Red': 0,
    'Yellow': 1,
    'Green': 2,
    'RedLeft': 0,
    'RedRight': 0,
    'RedStraight': 0,
    'RedStraightLeft': 0,
    'GreenLeft': 2,
    'GreenRight': 2,
    'GreenStraight': 2,
    'GreenStraightLeft': 2,
    'GreenStraightRight': 2,
    'off': 3
}
INVERSE_LABELS = {0: 'Red', 1: 'Yellow', 2: 'Green', 3: 'off'}
NUM_CLASSES = len(INVERSE_LABELS)


class BatchCallback(Callback):

    def __init__(self):
        super().__init__()
        self.training_sample_index = 0
        self.validation_sample_index = 0
        self.beginning_epoch = True

    def on_epoch_end(self, epoch, logs=None):
        self.training_sample_index = 0
        self.validation_sample_index = 0
        self.beginning_epoch = True


class InceptionV3TL:

    def __init__(self, args):
        self.args = args
        self.batch_callback = BatchCallback()

    def build_model(self):
        # download the InceptionV3 model weights based on ImageNet
        # don't include last few layers
        base_model = InceptionV3(weights='imagenet', include_top=False,
                                 input_tensor=Input(
                                     shape=(HEIGHT, WIDTH, CHANNELS)))

        # freeze all the convolutional layers
        for layer in base_model.layers:
            layer.trainable = False

        # construct the dense layers
        # apply dropout, 1024 neuron dense layer and softmax output layer
        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(units=1024, activation='elu')(x)

        # softmax converts a real vector to a vector of categorical
        # probabilities sums to 1
        predictions = Dense(units=NUM_CLASSES, activation='softmax')(x)

        # construct the model from the base model input and softmax layer as
        # output
        model = Model(inputs=base_model.input, outputs=predictions)

        # print the model summary
        model.summary()

        return model

    def load_training_data(self):
        datasets = [
            pd.read_csv(dataset_path)
            for dataset_path in self.args.dataset_paths
        ]

        x, y = [], []
        for dataset in datasets:
            x.extend(dataset['image_path'].values)
            y.extend(dataset['label'].values)

        x = np.array(x)
        y = np.array(y)

        x_train, x_val, y_train, y_val = \
            train_test_split(x, y,
                             test_size=self.args.split_size,
                             random_state=0,
                             shuffle=True)

        # create the one hot encoder object where the categories are set
        # automatically - unknown angles are ignored from the dataset
        encoder = OneHotEncoder(sparse=False)

        def labels_to_ints(labels):
            for i, y in enumerate(labels):
                labels[i] = LABELS[y]

            return labels

        y_train = labels_to_ints(y_train)
        y_val = labels_to_ints(y_val)

        # e.g. Red = 0 = [1, 0, 0, 0] when encoded
        y_train = list(encoder.fit_transform(y_train.reshape(-1, 1)))
        y_val = list(encoder.fit_transform(y_val.reshape(-1, 1)))

        return x_train, x_val, y_train, y_val

    def random_flip(self, image):
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)  # flip along vertical axis

        return image

    def random_shadow(self, image):
        height, width, channels = image.shape

        # create 2 points
        x1, y1 = width * np.random.rand(), 0
        x2, y2 = width * np.random.rand(), height
        xm, ym = np.mgrid[0:height, 0:width]

        # mask of zeros with same shape as one of image channels
        mask = np.zeros_like(image[:, :, 1])

        # find all the indexes that match a condition and give them a value
        # of 1 in the matrix (white)
        mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

        # create a boolean matrix where the mask elements is equal to
        # either 0 or 1
        cond = mask == np.random.randint(2)

        # draw samples from a uniform distribution between 0.2 and 0.5
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        # convert the colour image to HLS colour space
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

        # apply the ratio to the indexes of the L channel of the HLS colour
        # space that satisfy the condition of them either being 0 or 1
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio

        # convert back to RGB
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    def random_brightness(self, image):
        # convert the image to the HSV colours space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # random ratio for the brightness changes
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)

        # apply the ratio to the V channel of the HSV colours space which
        # represents the brightness value
        hsv[:, :, 2] = hsv[:, :, 2] * ratio

        # convert back to RGB before returning
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def data_augmentation(self, image):
        image = self.random_flip(image)  # random flip
        image = self.random_shadow(image)  # random shadow
        image = self.random_brightness(image)  # random brightness

        return image

    def pre_process_image(self, image):
        # resize
        image = cv2.resize(image, (WIDTH, HEIGHT), cv2.INTER_AREA)

        # inception preprocess
        image = preprocess_input(image)

        return image

    def batch_generator(self, image_paths, encoded_labels, training):
        batch_size = self.args.batch_size

        images = np.empty([batch_size, HEIGHT, WIDTH, CHANNELS])
        labels = np.empty([batch_size, NUM_CLASSES])

        while True:
            # matrices index
            i = 0

            if training and self.batch_callback.beginning_epoch:
                # shuffle before every epoch so different variations of
                # batches are selected
                permutation = np.random.permutation(len(image_paths))
                image_paths = np.asarray(image_paths)[permutation]
                encoded_labels = np.asarray(encoded_labels)[permutation]
                self.batch_callback.beginning_epoch = False

            if training:
                index_start = self.batch_callback.training_sample_index
                index_end = self.batch_callback.training_sample_index + \
                    batch_size
            else:
                index_start = self.batch_callback.validation_sample_index
                index_end = self.batch_callback.validation_sample_index + \
                    batch_size

            for index in range(index_start, index_end):
                try:
                    image_path = image_paths[index]
                    encoded_label = encoded_labels[index]
                except IndexError:
                    break

                image = cv2.imread(image_path, 1)

                # if training and np.random.rand() < 0.6:
                #     image = self.data_augmentation(image)

                image = self.pre_process_image(image)

                images[i] = np.array(image)
                labels[i] = np.array(encoded_label)

                i += 1

            if training:
                self.batch_callback.training_sample_index += batch_size
            else:
                self.batch_callback.validation_sample_index += batch_size

            # yield to CNN
            yield images, labels

    def train(self):
        model = self.build_model()

        # minimise the categorical cross-entropy via gradient descent using
        # Adam optimiser
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.args.learning_rate),
                      metrics=['accuracy'])

        x_train, x_val, y_train, y_val = \
            self.load_training_data()

        training_generator = \
            self.batch_generator(x_train, y_train, training=True)
        validation_generator = \
            self.batch_generator(x_val, y_val, training=False)

        training_iterations = int(len(x_train) / self.args.batch_size)
        validation_iterations = int(len(x_val) / self.args.batch_size)

        print(f'Batch Size: {self.args.batch_size}')
        print(f'Training: Amount - {len(x_train)}, Steps - {training_iterations}')
        print(f'Validation: Amount - {len(x_val)}, Steps - {validation_iterations}')

        model.fit_generator(
            training_generator,
            steps_per_epoch=training_iterations,
            epochs=self.args.epochs,
            max_queue_size=1,
            validation_data=validation_generator,
            validation_steps=validation_iterations,
            callbacks=[
                self.batch_callback,
                TensorBoard('logs'),
                ModelCheckpoint('model-{epoch:03d}.h5',
                                save_weights_only=True, period=5),
                EarlyStopping(monitor='val_loss', min_delta=0, patience=2,
                              verbose=0, mode='auto',
                              restore_best_weights=True)
            ],
            verbose=1
        )

        model.save('tl_model.h5')

    def make_prediction(self, model, image_path):
        image = cv2.imread(image_path, 1)
        image = self.pre_process_image(image)
        image = np.array([image])

        prediction_probs = model.predict(image)
        max_index = np.argmax(prediction_probs[0])
        predicted_label = INVERSE_LABELS[max_index]

        return predicted_label

    def test(self):
        model = self.build_model()
        model.load_weights(self.args.model_path)

        if self.args.dataset_path:
            dataset = pd.read_csv(self.args.dataset_path)

            num_correct = 0
            for index, row in dataset.iterrows():
                image_path = row['image_path']
                groundtruth_label = row['label']
                groundtruth_label = INVERSE_LABELS[LABELS[groundtruth_label]]

                predicted_label = self.make_prediction(model, image_path)
                print(groundtruth_label, predicted_label)

                if groundtruth_label == predicted_label:
                    num_correct += 1

            accuracy = (num_correct * 100) / len(dataset)
            print('Accuracy: ', accuracy)
        elif self.args.image_path:
            predicted_label = self.make_prediction(model, self.args.image_path)
            image = cv2.imread(self.args.image_path, 1)
            print('Traffic Light: ', predicted_label)
            cv2.imshow('Traffic Light', image)
            cv2.waitKey(0)
        elif self.args.api:
            pass


def main(args):
    nn = InceptionV3TL(args)

    if args.run_type == 'train':
        nn.train()
    elif args.run_type == 'test':
        nn.test()


def lst_type(s):
    return s.split(',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('train')
    parser_1.add_argument('dataset_paths', type=lst_type)
    parser_1.add_argument('--split_size', default=0.2)
    parser_1.add_argument('--learning_rate', default=1e-4)
    parser_1.add_argument('--batch_size', default=40)
    parser_1.add_argument('--epochs', default=10, type=int)

    parser_2 = sub_parsers.add_parser('test')
    parser_2.add_argument('model_path')
    parser_2.add_argument('--dataset_path')
    parser_2.add_argument('--image_path')
    parser_2.add_argument('--api')

    main(parser.parse_args())
