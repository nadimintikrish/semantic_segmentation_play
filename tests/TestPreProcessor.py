from data_preprocessor.PreProcessImgData import PreProcessImgData
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
'''
test label to array
'''

print("Checked Categorized label shape {}"
    .format(
    p.label_to_array(np.array(p.load(labels[1]))).shape
))

samples = p.file_list_samples[:10]

print(labels)
print(samples)

'''
test generator
'''
gen = p.train_generator(samples, labels, 2)

cat_img, cat_labels = next(gen)

print(cat_labels.shape)

img = p.reduce_to_img(cat_labels[1])

from PIL import Image

Image.fromarray(img.astype('uint8')).show()
Image
