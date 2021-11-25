import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob


width = 40
height = 60
num_images = 200
num_classes = 2
train_size = 150
test_size = num_images - train_size


def prepare_fei(in_path, out_path):
    all = np.ndarray(shape=(0, width*height+1))
    for file in glob.glob(in_path + '*.png8'):
        file_name = file.split('/')[-1]
        file_index = file_name[:-6]
        file_label = file_name[-6:-5]
        file_class = 0 if file_label == 'a' else 1
        img = Image.open(file)
        img_np = np.asarray(img)
        img_np = img_np.reshape((1, width*height))
        img_np = img_np.astype("float32") / 255.0
        file_class = np.array([file_class])
        file_class = file_class.reshape((1,1))
        img_np = np.append(img_np, file_class, axis=1)
        all = np.append(all, img_np, axis=0)

    fei_train, fei_test = train_test_split(all, test_size=test_size, random_state=1, shuffle=True)

    np.savetxt(out_path + 'train.txt', fei_train)
    np.savetxt(out_path + 'test.txt', fei_test)

    print('done')



in_path = '../data/FEI_1_images/'
out_path = '../data/fei_1/fei_1_'
prepare_fei(in_path, out_path)


print('done')
