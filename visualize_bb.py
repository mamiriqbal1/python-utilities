import numpy as np
from PIL import Image, ImageDraw

total_images = 1984
img_width = 28
resize_factor = 10
# base_path = '../XCS-IMG/cmake-build-debug/output-2-digit/2-digits-14/'
# image_file_path = "../XCS-IMG/data/mnist/mnist_validation_0_6.txt"
# visualization_file_path: str = base_path + 'visualization.txt'
# filter_file_path = base_path + '940785/filter.txt'
base_path = '../remote/output/XCS-IMG/output-2-digit/2-digits-02/879824/'
filter_file_path = base_path + '/filter.txt'
cf_file_path = base_path + '/code_fragment.txt'
cl_file_path = base_path + '/classifier.txt'

def get_blank_image(val):
    data = np.zeros((img_width, img_width))
    data += val
    return data


cl_data = {}

def load_cl_data():
    f = open(cl_file_path)
    line = f.readline()
    while line:
        tokens = line.strip().split()
        cl_id = int(tokens[1])
        fitness = float(tokens[9])
        num = int(tokens[3])
        exp = int(tokens[5])
        accuracy = float(tokens[11])
        prediction = float(tokens[13])
        error = float(tokens[15])
        action = int(tokens[21])

        line = f.readline()
        tokens = line.strip().split()
        cf_list = []
        for item in tokens:
            id = int(item)
            if id != -1:
                cf_list.append(id)
        cl_data[cl_id] = (cl_id, action, fitness, num, exp, accuracy, prediction, error, cf_list)
        line = f.readline()


load_cl_data()
cl_data_sorted = sorted(list(cl_data.values()), key=lambda tup: tup[2], reverse=True)
cl_data_sorted = np.array(cl_data_sorted)

cf_data = {}
def load_cf_data():
    f = open(cf_file_path)
    line = f.readline()
    while line:
        tokens = line.strip().split()
        cf_id = int(tokens[0])
        cf_num = int(tokens[1])
        cf_fit = int(tokens[2])
        cf_x = int(tokens[3])
        cf_y = int(tokens[4])
        cf_size_x = int(tokens[5])
        cf_size_y = int(tokens[6])
        filter_attributes = {}
        for i in range(7, len(tokens)):
            item = tokens[i];
            if(item.startswith("D")):
                filter_id = int(item[1:])
                filter_x = int(tokens[i+1])
                filter_y = int(tokens[i+2])
                filter_attributes[filter_id] = filter_x, filter_y
        line = f.readline()
        cf_data[cf_id] = (cf_id, cf_num, cf_fit, cf_x, cf_y, cf_size_x, cf_size_y, filter_attributes)

load_cf_data()
cf_data_sorted = sorted(list(cf_data.values()), key=lambda tup: tup[2], reverse=True)
cf_data_sorted = np.array(cf_data_sorted)


filter_data = {}
def load_filter_data():
    dilated = False
    f = open(filter_file_path)
    line = f.readline()
    while line:
        tokens = line.strip().split()
        f_id = int(tokens[1])
        size_x = int(tokens[3])
        size_y = int(tokens[5])
        dilated = bool(int(tokens[7]))
        line = f.readline()
        tokens = line.strip().split()
        values = []
        for i in range(size_x*size_y+1):
            if i == 0:  # skip the first string
                continue
            values.append(float(tokens[i]))
        line = f.readline()
        filter_data[f_id] = (values, size_x, size_y, dilated)


load_filter_data()


# update lower and upper bounds from filter
def update_bounds(img, values, start_x, start_y, size_x, size_y, dilated):
    for y in range(size_y):
        for x in range(size_x):
            img[start_y+y, start_x+x] = values[y*size_x + x]


def get_pixel_color(img, x, y):
    if img[y, x] == -1:  # if the pixel interval has not be initialized then its don't care
        return '#FF0000'  # "#7DCEA0"  # "#0000ff"
    c = int(img[y, x]*255)
    color = (c, c, c)
    return color


def visualize_intervals(img, dc):
    for y in range(img_width):
        for x in range(img_width):
            dc.point((x, y), get_pixel_color(img, x,y))


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def visualize_cf(cf_id_list):
    # initialize bounds to see if they are updated lower = 1, upper = 0
    img = get_blank_image(-1)
    base_img = Image.new("RGB", (img_width, img_width), "#000000")
    dc = ImageDraw.Draw(base_img)  # draw context
    print("Code Fragments: ", end=" ")
    for cf_id in cf_id_list:
        print(str(cf_id), end=" ")
        cf_id, cf_num, cf_fit, cf_x, cf_y, cf_size_x, cf_size_y, filter_attributes = cf_data[cf_id]
        for filter_item in filter_attributes:
            filter = filter_item
            x = cf_x + filter_attributes[filter][0]
            y = cf_y + filter_attributes[filter][1]
            values, size_x, size_y, dilated = filter_data[filter]
            update_bounds(img, values, x, y, size_x, size_y, dilated)

    print("")
    print("Total: " + str(len(cf_id_list)))
    visualize_intervals(img, dc)
    base_img = base_img.resize((img_width * resize_factor, img_width * resize_factor))
    base_img.show()
    input("press any key to continue")


def visualize_cl(cl_id_list):
    cf_id_list = []
    for cl_id in cl_id_list:
        for cf_id in cl_data[cl_id][8]:
            cf_id_list.append(cf_id)
    visualize_cf(cf_id_list)


def visualize_promisinng_cf():
    for cf in cf_data_sorted[:, 0]:
        cf_id = int(cf)
        visualize_cf([cf_id])


def visualize_promising_cl():
    for cl in cl_data_sorted[:, 0:2]:
        cl_id = int(cl[0])
        print("Classifier: " + str(cl_id) + " Class: " + str(cl[1]))
        visualize_cl([cl_id])


def visualize_all_cl(action):
    print('Visualizing all classifiers for action: ' + str(action))
    cl_id_list = []
    for cl in cl_data_sorted[:, 0:2]:
        if cl[1] == action:
            cl_id = int(cl[0])
            cl_id_list.append(cl_id)
    visualize_cl(cl_id_list)


# visualize_promisinng_cf()
# visualize_promising_cl()
visualize_all_cl(0)

print('done')
