import os
import glob
import random
import json
import pickle
import gc

import numpy as np
import cv2

from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.layers import concatenate
from keras.layers.core import Lambda
from keras.models import Model
from keras.utils.data_utils import Sequence
from keras import backend as K

import tensorflow as tf
        
class MapillaryGenerator(Sequence):
    def __init__(self, folder='datasets/mapillary', mode='training', n_classes=66, batch_size=1, resize_shape=None, 
                 crop_shape=(640, 320), horizontal_flip=True, vertical_flip=False, brightness=0.1, rotation=5, zoom=0.1):

        self.image_path_list = sorted(glob.glob(os.path.join(folder, mode, 'images/*')))
        self.label_path_list = sorted(glob.glob(os.path.join(folder, mode, 'instances/*')))
        self.mode = mode
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness = brightness
        self.rotation = rotation
        self.zoom = zoom
        
        # Preallocate memory
        if mode == 'training' and self.crop_shape:
            self.X = np.zeros((batch_size, crop_shape[1], crop_shape[0], 3), dtype='float32')
            self.Y1 = np.zeros((batch_size, crop_shape[1]//4, crop_shape[0]//4, self.n_classes), dtype='float32')
            self.Y2 = np.zeros((batch_size, crop_shape[1]//8, crop_shape[0]//8, self.n_classes), dtype='float32')
            self.Y3 = np.zeros((batch_size, crop_shape[1]//16, crop_shape[0]//16, self.n_classes), dtype='float32')
        elif self.resize_shape:
            self.X = np.zeros((batch_size, resize_shape[1], resize_shape[0], 3), dtype='float32')
            self.Y1 = np.zeros((batch_size, resize_shape[1]//4, resize_shape[0]//4, self.n_classes), dtype='float32')
            self.Y2 = np.zeros((batch_size, resize_shape[1]//8, resize_shape[0]//8, self.n_classes), dtype='float32')
            self.Y3 = np.zeros((batch_size, resize_shape[1]//16, resize_shape[0]//16, self.n_classes), dtype='float32')
        else:
            raise Exception('No image dimensions specified!')
        
    def __len__(self):
        return len(self.image_path_list) // self.batch_size
        
    def __getitem__(self, i):        
        for n, (image_path, label_path) in enumerate(zip(self.image_path_list[i*self.batch_size:(i+1)*self.batch_size], 
                                                        self.label_path_list[i*self.batch_size:(i+1)*self.batch_size])):

            image = cv2.imread(image_path, 1)
            label = cv2.imread(label_path, 0)
            if self.resize_shape:
                image = cv2.resize(image, self.resize_shape)
                label = cv2.resize(label, self.resize_shape)

            # Do augmentation (only if training)
            if self.mode == 'training':
                if self.horizontal_flip and random.randint(0,1):
                    image = cv2.flip(image, 1)
                    label = cv2.flip(label, 1)
                if self.vertical_flip and random.randint(0,1):
                    image = cv2.flip(image, 0)
                    label = cv2.flip(label, 0)
                if self.brightness:
                    factor = 1.0 + abs(random.gauss(mu=0, sigma=self.brightness))
                    if random.randint(0,1):
                        factor = 1.0/factor
                    image = (255.0*((image/255.0)**factor)).astype(np.uint8)
                if self.rotation:
                    angle = random.gauss(mu=0.0, sigma=self.rotation)
                else:
                    angle = 0.0
                if self.zoom:
                    scale = random.gauss(mu=1.0, sigma=self.zoom)
                else:
                    scale = 1.0
                if self.rotation or self.zoom:
                    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, scale)
                    image = cv2.warpAffine(image, M, image.shape[:2])
                    label = cv2.warpAffine(label, M, label.shape[:2])
                if self.crop_shape:
                    image, label = _random_crop(image, label, self.crop_shape)

            self.X[n] = image
            self.Y1[n] = to_categorical(cv2.resize(label, (label.shape[1]//4, label.shape[0]//4)), self.n_classes).reshape((label.shape[0]//4, label.shape[1]//4, -1))   
            self.Y2[n] = to_categorical(cv2.resize(label, (label.shape[1]//8, label.shape[0]//8)), self.n_classes).reshape((label.shape[0]//8, label.shape[1]//8, -1))
            self.Y3[n] = to_categorical(cv2.resize(label, (label.shape[1]//16, label.shape[0]//16)), self.n_classes).reshape((label.shape[0]//16, label.shape[1]//16, -1))         
        
        return self.X, [self.Y1, self.Y2, self.Y3]
        
    def on_epoch_end(self):
        # Shuffle dataset for next epoch
        c = list(zip(self.image_path_list, self.label_path_list))
        random.shuffle(c)
        self.image_path_list, self.label_path_list = zip(*c)

        # Fix memory leak (Keras bug)
        gc.collect()
                
class Visualization(Callback):
    def __init__(self, resize_shape=(640, 320), batch_steps=10, n_gpu=1, **kwargs):
        super(Visualization, self).__init__(**kwargs)
        self.resize_shape = resize_shape
        self.batch_steps = batch_steps
        self.n_gpu = n_gpu
        self.counter = 0

        # TODO: Remove this lazy hardcoded paths
        self.test_images_list = glob.glob('datasets/mapillary/testing/images/*')
        with open('datasets/mapillary/config.json') as config_file:
            config = json.load(config_file)
        self.labels = config['labels']
        
        
    def on_batch_end(self, batch, logs={}):
        self.counter += 1
        
        if self.counter == self.batch_steps:
            self.counter = 0
            
            test_image = cv2.resize(cv2.imread(random.choice(self.test_images_list), 1), self.resize_shape)
            
            inputs = [test_image]*self.n_gpu          
            output, _, _ = self.model.predict(np.array(inputs), batch_size=self.n_gpu)
        
            cv2.imshow('input', test_image)
            cv2.waitKey(1)
            cv2.imshow('output', apply_color_map(np.argmax(output[0], axis=-1), self.labels))
            cv2.waitKey(1)

class PolyDecay:
    def __init__(self, initial_lr, power, n_epochs):
        self.initial_lr = initial_lr
        self.power = power
        self.n_epochs = n_epochs
    
    def scheduler(self, epoch):
        return self.initial_lr * np.power(1.0 - 1.0*epoch/self.n_epochs, self.power)
            
class ExpDecay:
    def __init__(self, initial_lr, decay):
        self.initial_lr = initial_lr
        self.decay = decay
    
    def scheduler(self, epoch):
        return self.initial_lr * np.exp(-self.decay*epoch)
    
# Taken from Mappillary Vistas demo.py
def apply_color_map(image_array, labels):
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    for label_id, label in enumerate(labels):
        # set all pixels with the current label to the color of the current label
        color_array[image_array == label_id] = label["color"]

    return color_array
    
def _random_crop(image, label, crop_shape):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
        x = random.randrange(image.shape[1]-crop_shape[0])
        y = random.randrange(image.shape[0]-crop_shape[1])
        
        return image[y:y+crop_shape[1], x:x+crop_shape[0], :], label[y:y+crop_shape[1], x:x+crop_shape[0]]
    else:
        raise Exception('Crop shape exceeds image dimensions!')
