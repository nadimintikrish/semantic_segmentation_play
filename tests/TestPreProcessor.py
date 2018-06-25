from data_preprocessor.PreProcessImgData import PreProcessImgData
import numpy as np
from PIL import Image

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

labels_dict = [[0, 0, 0], [128, 0, 0], [128, 64, 128]]

p = PreProcessImgData(image_path_samples,
                      image_path_labels,
                      3, 256, 256, labels_dict)

labels = p.file_list_labels[:100]
samples = p.file_list_samples[:100]

labels = sorted(labels)
samples = sorted(samples)

# from sklearn.model_selection import train_test_split
#
# train_samples, test_samples, train_labels, test_labels = \
#     train_test_split(p.file_list_samples, p.file_list_labels, test_size=0.33, random_state=42)
#
# test_lbl = p.label_to_array(np.array(p.load(train_labels[245])))
#
# Image.fromarray(p.reduce_to_img(test_lbl).astype('uint8')).show()

'''
test label to array
'''

# print("Checked Categorized label shape {}"
#     .format(
#     p.label_to_array(np.array(p.load(labels[1]))).shape
# ))
#
# samples = p.file_list_samples[:10]
#
# print(labels)
# print(samples)

'''
test generator
'''
gen = p.train_generator(samples, labels, 2, test=False)

cat_img, cat_labels = next(gen)

print(cat_img[0])

# img = p.reduce_to_img(cat_labels[1])
#
# from PIL import Image
#
# Image.fromarray(img.astype('uint8')).show()
