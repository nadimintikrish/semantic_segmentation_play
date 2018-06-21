import glob

import numpy as np
from keras.utils import to_categorical


class PreProcessImgData:

    def __init__(self, img_path_samples, img_path_labels, num_classes):
        self.file_list_samples = glob.glob(img_path_samples)
        self.file_list_labels = glob.glob(img_path_labels)
        self.num_classes = num_classes
        self.labels_dict = None  ## to be hardcoded
        self.train_samples_array = None
        self.test_samples_array = None
        self.train_labels_array = None
        self.test_labels_array = None

    def normalize_tensors(self):
        self.train_samples_array = self.train_samples_array.astype('float32') / 255
        self.test_samples_array = self.test_samples_array.astype('float32') / 255
        self.train_labels_array = self.train_labels_array.astype('float32') / 255
        self.test_labels_array = self.test_labels_array.astype('float32') / 255

    def reduce_for_categorical(self, label):
        return np.zeros(self.num_classes).reshape(label.shape[0], label.shape[1], 1), label[0], label[1]

    '''
    converts a label of form (w*h*3) to (w*h*num_classes)
    '''

    def label_to_array(self, label_array):
        cat_zero_array, w, h = self.reduce_for_categorical(label_array)
        for i in range(w):
            for j in range(h):
                cat_zero_array[i][j] = \
                    self.labels_dict.index(label_array[i][j].tolist())

        return to_categorical(cat_zero_array)
