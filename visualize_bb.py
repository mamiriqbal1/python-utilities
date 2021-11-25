import numpy as np
from PIL import Image, ImageDraw

# total_images = 1984
img_width = 28
img_height = 28
resize_factor = 10
base_path = '../XCS-IMG/cmake-build-release/output-mnist/mnist-02/60000'
# base_path = '../remote/output/XCS-IMG/output-10-digit/10-digits-07/3800000'
filter_file_path = base_path + '/filter.txt'
cf_file_path = base_path + '/code_fragment.txt'
cl_file_path = base_path + '/classifier.txt'
max_condition_length = 25
visualization_file_path = base_path + '/visualization.txt'

image_file_path = '../XCS-IMG/data/mnist/mnist_test.txt'
# image_file_path = '../XCS-IMG/data/fei_1/fei_1_train.txt'

img_file = np.loadtxt(image_file_path)


def get_image(img_id, denormalize):
    item = img_file[img_id]
    img_class = int(item[-1])
    data = item[:-1]
    # denormalize
    if denormalize:
        data = data * 255
    data = data.reshape(28, 28)
    return img_class, data


def get_blank_image(val):
    data = np.zeros((img_height, img_width))
    data += val
    return data


cl_data = {}

def load_cl_data():
    f = open(cl_file_path)
    line = f.readline()
    line = f.readline()
    while line:
        tokens = line.strip().split()
        cl_id = int(tokens[0])
        cf_count = int(tokens[3])
        fitness = float(tokens[4])
        num = int(tokens[1])
        exp = int(tokens[2])
        accuracy = float(tokens[5])
        prediction = float(tokens[6])
        error = float(tokens[7])
        action = int(tokens[10])

        cf_list = []
        for i in range(max_condition_length):
            id = int(tokens[i+11])
            if id != -1:
                cf_list.append(id)
        cl_data[cl_id] = (cl_id, action, fitness, num, exp, accuracy, prediction, error, cf_list)
        line = f.readline()


load_cl_data()
cl_data_sorted = sorted(list(cl_data.values()), key=lambda tup: tup[2], reverse=True)
cl_data_sorted = np.array(cl_data_sorted, dtype=object)

cf_data = {}
def load_cf_data():
    f = open(cf_file_path)
    line = f.readline()
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

        pattern = []
        pattern_start_index = 8
        for y in range(cf_size_y):
            row = []
            for x in range(cf_size_x):
                row.append(float(tokens[pattern_start_index + y*cf_size_y + x]));
            pattern.append(row)

        line = f.readline()
        cf_data[cf_id] = (cf_id, cf_num, cf_fit, cf_x, cf_y, cf_size_x, cf_size_y, pattern)

load_cf_data()
cf_data_sorted = sorted(list(cf_data.values()), key=lambda tup: tup[2], reverse=True)
cf_data_sorted = np.array(cf_data_sorted, dtype=object)


visualization_data = {}


def load_visualization_data():
    # load visualization data
    actual_class = -1
    predicted_class = -1
    f = open(visualization_file_path)
    line = f.readline()
    while line:
        tokens = line.strip().split()
        read_img_id = int(tokens[0])
        actual_class = int(tokens[1])
        predicted_class = int(tokens[2])
        line = f.readline()  # read list of action set classifiers
        tokens = line.strip().split()
        cl_ids = []
        for id in tokens:
            cl_ids.append(int(id))
        visualization_data[read_img_id] = (actual_class, predicted_class, cl_ids)
        line = f.readline()
    f.close()


load_visualization_data()


def update_pixel_data(img_sum, img_count, cf_id):
    cf_id, cf_num, cf_fit, cf_x, cf_y, cf_size_x, cf_size_y, pattern = cf_data[cf_id]
    for y in range(cf_size_y):
        for x in range(cf_size_x):
            img_sum[cf_y + y, cf_x + x] += pattern[y][x]
            img_count[cf_y + y, cf_x + x] += 1


def get_pixel_color(img_sum, img_count, x, y):
    if img_count[y, x] == 0:
        return '#FF0000'
    c = int(img_sum[y, x]/img_count[y, x] * 255)
    color = (c, c, c)
    return color


def visualize_intervals(img_sum, img_count, dc):
    for y in range(img_height):
        for x in range(img_width):
            dc.point((x, y), get_pixel_color(img_sum, img_count, x, y))


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def visualize_cf(cf_id_list, original_img=None):
    print('Code Fragments: (' + str(len(cf_id_list)) + ') ' + str(cf_id_list))
    # initialize bounds to see if they are updated lower = 1, upper = 0
    img_sum = get_blank_image(0)
    img_count = get_blank_image(0)
    base_img = Image.new("RGB", (img_width, img_height), "#000000")
    dc = ImageDraw.Draw(base_img)  # draw context
    for cf_id in cf_id_list:
        cf_id, cf_num, cf_fit, cf_x, cf_y, cf_size_x, cf_size_y, pattern = cf_data[cf_id]
        update_pixel_data(img_sum, img_count, cf_id)

    visualize_intervals(img_sum, img_count, dc)
    base_img = base_img.resize((img_width * resize_factor, img_height * resize_factor))
    all_img = base_img
    if original_img is not None:
        all_img = get_concat_h(base_img, original_img)
    all_img.show()
    input("press any key to continue")


def visualize_cl(cl_ids, original_img=None):
    print('Classifiers: (' + str(len(cl_ids)) + ') ' + str(cl_ids))
    cf_id_list = []
    for cl_id in cl_ids:
        # if promising and cl_data[cl_id][7] >= 10 or cl_data[cl_id][4] < 10:
        # if promising and cl_data[cl_id][2] < 0.01:
        #     continue
        for cf_id in cl_data[cl_id][8]:
            cf_id_list.append(cf_id)
    visualize_cf(cf_id_list, original_img)


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


def filter_cls(cl_ids):
    filtered_ids = []
    for cl_id in cl_ids:
        # if promising and cl_data[cl_id][7] >= 10 or cl_data[cl_id][4] < 10:
        if cl_data[cl_id][2] >= 0.1:
            filtered_ids.append(cl_id)
    return filtered_ids


def get_original_image(img_id):
    digit, image = get_image(img_id, True)
    image = np.copy(image)
    original_image = Image.fromarray(image).convert("RGB")
    original_image = original_image.resize((img_width * resize_factor, img_height* resize_factor))
    return original_image


def visualize_action_set():
    print("Visualizing action set for validation images")
    for img_id in visualization_data.keys():
        actual_class, predicted_class, cl_ids = visualization_data[img_id]
        print('Image: ' + str(img_id) + ' Actual: ' + str(actual_class) + ' Predicted: ' + str(predicted_class))
        # cl_ids = filter_cls(cl_ids)
        visualize_cl(cl_ids, get_original_image(img_id))


def show_original_image(img_id):
    img = get_original_image(img_id)
    img.show()


# visualize_cl([1892])
# visualize_promisinng_cf()
# visualize_promising_cl()
# visualize_all_cl(1)
visualize_action_set()
# visualize_cf([3868])
# show_original_image(9107)

print('done')
