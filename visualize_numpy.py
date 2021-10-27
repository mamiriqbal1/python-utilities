from PIL import Image
import numpy as np

height = 7
width = 7
resize_factor = 30
base_path = '../autoencoder/data/mnist/'
file_name = 'test_encoded_mnist.txt'
data = np.loadtxt(base_path + file_name)

for i in range(data.shape[0]):
    data_2d = np.reshape(data[i], (height, width))
    img = Image.fromarray(data_2d, 'RGB')
    img = img.resize((width*resize_factor, height*resize_factor))
    img.show()
    input("press any key to continue")
