from PIL import Image
import numpy as np

height = 7
width = 7
resize_factor = 30
base_path = '../XCS-IMG/data/mnist/'
file_name = 'mnist_train_encoded7x7.txt'
data = np.loadtxt(base_path + file_name)
img = data[:, :-1]
label = data[:, -1]
show_digit = 0

for i in range(img.shape[0]):
    if show_digit != -1 and label[i] != show_digit:
        continue
    data_2d = np.reshape(img[i], (height, width))
    img_2 = Image.fromarray(data_2d, 'RGB')
    img_2 = img_2.resize((width*resize_factor, height*resize_factor))
    img_2.show()
    input("press any key to continue " + str(label[i]))
