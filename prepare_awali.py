import numpy as np
from sklearn.model_selection import train_test_split

in_path = '../data/awali/';
out_path = '../data/awali/';


def prepare_dataset(in_path, out_path, file_name):
    digit = np.loadtxt(in_path + file_name, delimiter=',')
    digit_train, digit_test = train_test_split(digit)
    np.savetxt(out_path + 'train_' + file_name, digit_train, fmt='%1d')
    np.savetxt(out_path + 'test_' + file_name, digit_test, fmt='%1d')
    print('done')

prepare_dataset(in_path, out_path, 'devaDigit.csv')

