import glob
from random import randrange

import numpy as np
from keras.utils import to_categorical
from keras.preprocessing import image


class PreProcessImgData:

    def __init__(self, img_path_samples, img_path_labels, num_classes, height, width, labels_dict):
        self.file_list_samples = glob.glob(img_path_samples)
        self.file_list_labels = glob.glob(img_path_labels)
        self.num_classes = num_classes
        self.labels_dict = labels_dict
        self.height = height
        self.width = width
        self.void_class = 0

    '''
    converts a label of form (w*h*3) to (w*h*num_classes)
    '''

    def label_to_array(self, label_array):
        cat_zero_array = self.reduce_for_categorical(label_array)
        for i in range(self.width):
            for j in range(self.height):
                list_val = label_array[i][j].tolist()

                if list_val in self.labels_dict:
                    cat_zero_array[i][j] = \
                        self.labels_dict.index(list_val)
                else:
                    cat_zero_array[i][j] = self.void_class

        return to_categorical(cat_zero_array, self.num_classes)

    def load(self, path):
        return image.load_img(path, target_size=(self.width, self.height))

    def train_generator(self, samples, labels, batch_size):
        batch_samples = np.zeros((batch_size, self.height, self.width, 3))
        batch_labels = np.zeros((batch_size, self.height, self.width, self.num_classes))

        while True:
            for i in range(batch_size):
                index = randrange(len(samples))
                batch_samples[i] = np.array(self.load(samples[index]))
                batch_labels[i] = self.label_to_array(np.array(self.load(labels[index])))
            yield self.normalize_tensors(batch_samples), batch_labels

    @staticmethod
    def reduce_for_categorical(label):
        return np.zeros(label.shape[0] * label.shape[1]).reshape(label.shape[0], label.shape[1], 1)

    @staticmethod
    def normalize_tensors(tensors):

        return tensors.astype('float32') / 255
