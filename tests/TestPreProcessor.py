from data_preprocessor.PreProcessImgData import PreProcessImgData
from keras.preprocessing import image
import numpy as np

'''
Void		0 	0 	0
Building 	128 	0 	0
XXgrass		0 128 0

Tree		128 	128 	0
XXcow		0 0 128
XXhorse		128 0 128

'''

image_path_samples = 'C:/Krishna/DataSets/cam_vid_semantic/train_samples/*.png'
image_path_labels = 'C:/Krishna/DataSets/cam_vid_semantic/train_labels/*.png'

labels_dict = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128]]

p = PreProcessImgData(image_path_samples,
                      image_path_labels,
                      6, 128, 128, labels_dict)

labels = p.file_list_labels[:10]

print(p.label_to_array(np.array(p.load(labels[1]))).shape)
