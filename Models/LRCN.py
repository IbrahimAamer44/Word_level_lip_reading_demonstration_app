# Import the required libraries.
import os
import cv2
import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from jiwer import wer
from moviepy.editor import *

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model


def create_LRCN_model():
    SEQUENCE_LENGTH = 15
    IMAGE_HEIGHT = 50
    IMAGE_WIDTH = 100
    CLASSES_LIST = ['ہے', 'کیسے', 'چار', 'دو', 'چھے', 'وہ', 'جی', 'کب', 'پانچھ', 'تین', 'آپ', 'نوں', 'ہاں', 'میں',
                    'تھا', 'ہوں', 'نہیں', 'کیوں', 'کتنے', 'ایک', 'کون', 'تھے', 'ہم', 'آٹھ', 'کونسا', 'کدھر', 'سات']

    '''
    This function will construct the required LRCN model.
    Returns:
        model: It is the required constructed LRCN model.
    '''

    # We will use a Sequential model for model construction.
    model = Sequential()

    # Define the Model Architecture.
    ########################################################################################################################

    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='valid', activation='relu'),
                              input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='valid', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='valid', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='valid', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    # model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))

    model.add(GRU(64, return_sequences=True))
    model.add(GRU(64))

    # model.add(Bidirectional(GRU(32)))

    model.add(Dense(len(CLASSES_LIST), activation='softmax'))

    ########################################################################################################################
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=["accuracy"])
    # Display the models summary.
    #model.summary()

    # Return the constructed LRCN model.
    return model


if __name__ == "__main__":

    seed_constant = 27
    np.random.seed(seed_constant)
    random.seed(seed_constant)
    tf.random.set_seed(seed_constant)


    # Important variables DEFINED IN PREPROCESSOR FILES
    SEQUENCE_LENGTH = 15
    IMAGE_HEIGHT = 50
    IMAGE_WIDTH = 100
    CLASSES_LIST = ['ہے', 'کیسے', 'چار', 'دو', 'چھے', 'وہ', 'جی', 'کب', 'پانچھ', 'تین', 'آپ', 'نوں', 'ہاں', 'میں',
                    'تھا', 'ہوں', 'نہیں', 'کیوں', 'کتنے', 'ایک', 'کون', 'تھے', 'ہم', 'آٹھ', 'کونسا', 'کدھر', 'سات']
    model = create_LRCN_model()