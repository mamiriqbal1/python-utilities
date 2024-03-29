from keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_mnist_all(out_path):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # reshape
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # add labels as last column
    x_train = np.hstack((x_train, np.reshape(y_train, (-1, 1))))
    x_test = np.hstack((x_test, np.reshape(y_test, (-1, 1))))

    np.savetxt(out_path + 'mnist_train.txt', x_train, fmt='%1d')
    np.savetxt(out_path + 'mnist_test.txt', x_test, fmt='%1d')


def prepare_mnist_digits(out_path, digits, postfix):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # prepare the train set
    index_all = np.ndarray(shape=(0,), dtype=np.int64)
    for digit in digits:
        index_digit = np.where(y_train == digit)[0]
        index_all = np.concatenate((index_all, index_digit), axis=None)

    index_all = np.sort(index_all)
    y_train = y_train[index_all]
    x_train = x_train[index_all]

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

    for digit, label in zip(digits, range(len(digits))):
        y_train[y_train == digit] = label

    # prepare the test set
    index_all = np.ndarray(shape=(0,), dtype=np.int64)
    for digit in digits:
        index_digit = np.where(y_test == digit)[0]
        index_all = np.concatenate((index_all, index_digit), axis=None)

    index_all = np.sort(index_all)
    y_test = y_test[index_all]
    x_test = x_test[index_all]

    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    for digit, label in zip(digits, range(len(digits))):
        y_test[y_test == digit] = label

    # split training set by extracting same number of validation instances as test
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=y_test.size)

    # normalize between 0 and 1
    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    x_validation = (x_validation - x_validation.min()) / (x_validation.max() - x_validation.min())
    x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())


    np.savetxt(out_path + 'mnist_train' + postfix + '.txt', np.hstack((x_train, np.reshape(y_train, (-1, 1)))))
    np.savetxt(out_path + 'mnist_validation' + postfix + '.txt', np.hstack((x_validation, np.reshape(y_validation, (-1, 1)))))
    np.savetxt(out_path + 'mnist_test' + postfix + '.txt', np.hstack((x_test, np.reshape(y_test, (-1, 1)))))


def prepare_mnist_encoded(data_path, label_path, out_path):
    data = np.loadtxt(data_path)
    label = np.loadtxt(label_path)
    label = np.reshape(label, (label.shape[0], 1))
    encoded_mnist = np.concatenate((data, label), axis=1)
    np.savetxt(out_path, encoded_mnist)
    print('done')



height = 7
width = 7
in_base_path = '../autoencoder/data/mnist/'
data_file_name = 'train_encoded_mnist.txt'
label_file_name = 'train_label_mnist.txt'
data_path = in_base_path + data_file_name
label_path = in_base_path + label_file_name
out_base_path = '../XCS-IMG/data/mnist/'
out_file_name = 'mnist_train_encoded7x7.txt'
out_path = out_base_path + out_file_name
# prepare_mnist_encoded(data_path, label_path, out_path)

# prepare_mnist_digits('../data/mnist/', [0, 6], '_0_6')
# prepare_mnist_digits('../data/mnist/', [3, 8], '_3_8')
# prepare_mnist_digits('../data/mnist/', [3, 8, 5, 6], '_3_8_5_6')
# prepare_mnist_digits('../data/mnist/', [1, 2, 4, 7, 9, 0], '_1_2_4_7_9_0')
prepare_mnist_all('../data/mnist/')

#prepare_mnist_3_8('../data/mnist/')

# prepare_mnist_all('../data/mnist/')
