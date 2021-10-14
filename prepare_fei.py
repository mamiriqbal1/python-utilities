import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob


def prepare_fei(in_path, out_path):
    all = np.ndarray(shape=(0, 2400))
    for file in glob.glob(in_path + '*.png8'):
        img = Image.open(file)
        img_np = np.asarray(img)
        img_np = img_np.reshape((1,2400))
        all = np.append(all, img_np, axis=0)



in_path = '../data/FEI_1_images/'
out_path = '../data/FEI_1/'
prepare_fei(in_path, out_path)

print('done')
