import glob
from PIL import Image


class PreProcessImgData:

    def __init__(self, img_path_samples, img_path_labels):
        self.file_list_samples = glob.glob(img_path_samples)
        self.file_list_labels = glob.glob(img_path_labels)
        self.train_samples_array = None
        self.test_samples_array = None
        self.train_labels_array = None
        self.test_labels_array = None

    def normalize_tensors(self):
        self.train_samples_array = self.train_samples_array.astype('float32') / 255
        self.test_samples_array = self.test_samples_array.astype('float32') / 255
        self.train_labels_array = self.train_labels_array.astype('float32') / 255
        self.test_labels_array = self.test_labels_array.astype('float32') / 255

