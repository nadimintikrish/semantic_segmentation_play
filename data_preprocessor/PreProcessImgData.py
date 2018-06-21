import glob
from random import randrange

import numpy as np
from keras.utils import to_categorical
from keras.preprocessing import image


class PreProcessImgData:

    def __init__(self, img_path_samples, img_path_labels, num_classes, height, width):
        self.file_list_samples = glob.glob(img_path_samples)
        self.file_list_labels = glob.glob(img_path_labels)
        self.num_classes = num_classes
        self.labels_dict = None  ## to be hardcoded
        self.height = height
        self.width = width

    def reduce_for_categorical(self, label):
        return np.zeros(self.num_classes).reshape(label.shape[0], label.shape[1], 1), label[0], label[1]

    '''
    converts a label of form (w*h*3) to (w*h*num_classes)
    '''

    def label_to_array(self, label_array):
        cat_zero_array, w, h = self.reduce_for_categorical(label_array)
        for i in range(w):
            for j in range(h):
                list_val = label_array[i][j].tolist()
                if list_val in self.labels_dict:
                    cat_zero_array[i][j] = \
                        self.labels_dict.index(list_val)
                else:
                    cat_zero_array[i][j] = [0] * 3

        return to_categorical(cat_zero_array)

    def load(self, path):
        return image.load_img(path, target_size=(self.width, self.height))

    def train_generator(self, file_list_samples, file_list_labels, batch_size):
        batch_samples = np.zeros((batch_size, self.height, self.width, 3))
        batch_labels = np.zeros((batch_size, self.height, self.width, self.num_classes))

        while True:
            for i in range(batch_size):
                index = randrange(len(file_list_samples))
                batch_samples[i] = np.array(self.load(file_list_samples[index]))
                batch_labels[i] = self.label_to_array(np.array(self.load(file_list_labels[index])))

            yield batch_samples, batch_labels

    @staticmethod
    def normalize_tensors(train_samples_array, test_samples_array):

        return train_samples_array.astype('float32') / 255, \
               test_samples_array.astype('float32') / 255
