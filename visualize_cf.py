import numpy as np
from PIL import Image, ImageDraw

total_images = 1984
img_width = 28
resize_factor = 10
# base_path = '../XCS-IMG/cmake-build-debug/output-2-digit/2-digits-14/'
# image_file_path = "../XCS-IMG/data/mnist/mnist_validation_0_6.txt"
# visualization_file_path: str = base_path + 'visualization.txt'
# filter_file_path = base_path + '940785/filter.txt'
base_path = '../XCS-IMG/cmake-build-debug/output-4-digit/4-digits-18/'
iteration_number = '1636908'
image_file_path = "../XCS-IMG/data/mnist/mnist_validation_all.txt"
visualization_file_path: str = base_path + 'visualization.txt'
filter_file_path = base_path + iteration_number + '/filter.txt'
cf_file_path = base_path + iteration_number + '/code_fragment.txt'
cl_file_path = base_path + iteration_number + '/classifier.txt'


img_file = np.loadtxt(image_file_path)


def get_blank_image(val):
    data = np.zeros((img_width, img_width))
    data += val
    return data


def get_image(img_id, denormalize):
    item = img_file[img_id]
    img_class = int(item[-1])
    data = item[:-1]
    # denormalize
    if denormalize:
        data = data * 255
    data = data.reshape(28, 28)
    return img_class, data


cl_data = {}

def load_cl_data():
    f = open(cl_file_path)
    line = f.readline()
    while line:
        tokens = line.strip().split()
        cl_id = int(tokens[1])
        line = f.readline()
        tokens = line.strip().split()
        cf_list = []
        for item in tokens:
            id = int(item)
            if id != -1:
                cf_list.append(id)
        cl_data[cl_id] = cf_list
        line = f.readline()


load_cl_data()


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
        cf_size = int(tokens[5])
        filter_attributes = {}
        for i in range(6, len(tokens)):
            item = tokens[i];
            if(item.startswith("D")):
                filter_id = int(item[1:])
                filter_x = int(tokens[i+1])
                filter_y = int(tokens[i+2])
                filter_attributes[filter_id] = filter_x, filter_y
        line = f.readline()
        cf_data[cf_id] = (cf_id, cf_num, cf_fit, cf_x, cf_y, cf_size, filter_attributes)

load_cf_data()
cf_data_sorted = sorted(list(cf_data.values()), key=lambda tup: tup[2], reverse=True)
cf_data_sorted = np.array(cf_data_sorted)


filter_data = {}
def load_filter_data():
    f_id = -1
    size = -1
    dilated = False
    f = open(filter_file_path)
    line = f.readline()
    while line:
        tokens = line.strip().split()
        f_id = int(tokens[1])
        size = int(tokens[3])
        dilated = bool(int(tokens[5]))
        line = f.readline()
        tokens = line.strip().split()
        lb = []
        ub = []
        for i in range(size*size+1):
            if i == 0:  # skip the first string
                continue
            lb.append(float(tokens[i]))
        line = f.readline()
        tokens = line.strip().split()
        for i in range(size*size+1):
            if i == 0:
                continue
            ub.append(float(tokens[i]))
        line = f.readline()
        filter_data[f_id] = (lb, ub, size, dilated)


load_filter_data()


# update lower and upper bounds from filter
def update_bounds(img_l, img_u, lb, ub, start_x, start_y, size, dilated):
    step = 1
    if dilated:
        step = 2
        effective_size = size + size - 1

    # for y in range(size):
    #     for x in range(size):
    #         img_l[start_x+x*step, (start_y+y*step)] = .4
    #         img_u[start_x+x*step, (start_y+y*step)] = .6

    for y in range(size):
        for x in range(size):
            if img_l[start_x+x*step, (start_y+y*step)] < lb[y*size + x]:
                img_l[start_x+x*step, (start_y+y*step)] = lb[y*size + x]
            if img_u[start_x+x*step, (start_y+y*step)] > ub[y*size + x]:
                img_u[start_x+x*step, (start_y+y*step)] = ub[y*size + x]


def get_pixel_color(img_l, img_u, x, y, lower):
    if img_l[x, y] == -1 and img_u[x, y] == 2:  # if the pixel interval has not be initialized then its don't care
        return "#000000"  # "#7DCEA0"  # "#0000ff"
    if img_l[x, y] == 0 and img_u[x, y] == 1:  # if the pixel interval has max then its don't care
        return "#000000"  # "#7DCEA0"  #ff0000"  #"#006400"

    mid = (img_l[x,y] + img_u[x,y]) / 2
    # real to 255 scale
    c = int(mid*255)
    color = (c, c, c)
    return color

    if img_l[x, y] == 0:  # this interval accepts black (non-white, or grey to black)
        return "#000000"
    if img_u[x, y] == 1:  # this interval accepts white (non-black, or grey to  white)
        return "#ffffff"
    # if img_u[x, y] - img_l[x, y] > 0.5:  # wide interval means don't care
    #     return color
    # if img_l[x, y] < 0.25 and img_u[x, y] > 0.75:  # wide interval means don't care
    #     return color

    if lower:
        return "#BFBFBF"  # "#D35400"

    # if lower is true then return lower bound otherwise upper bound
    # if lower:
    #     mid = img_l[x,y]
    #     if mid == 0:
    #         return "#00ff00"
    # else:
    #     mid = img_u[x,y]
    #     if mid == 1:
    #         return "#00ff00"



def visualize_intervals(img_l, img_u, dc, lower):
    for y in range(img_width):
        for x in range(img_width):
            dc.point((x, y), get_pixel_color(img_l, img_u, x,y, lower))


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def visualize_cf():
    for cl_id in cl_data.keys():
        # initialize bounds to see if they are updated lower = 1, upper = 0
        img_l = get_blank_image(-1)
        img_u = get_blank_image(2)
        base_img = Image.new("RGB", (img_width, img_width), "#000000")
        dc = ImageDraw.Draw(base_img)  # draw context
        for cf_id in cl_data[cl_id]:
            # for cf in cf_data_sorted[:, 0]:
            #     cf_id = int(cf)

            cf_id, cf_num, cf_fit, cf_x, cf_y, cf_size, filter_attributes = cf_data[cf_id]
            for filter_item in filter_attributes:
                filter = filter_item
                x = cf_x + filter_attributes[filter][0]
                y = cf_y + filter_attributes[filter][1]
                lbf, ubf, size, dilated = filter_data[filter]
                lb = lbf.copy()
                ub = ubf.copy()
                update_bounds(img_l, img_u, lb, ub, x, y, size, dilated)

        visualize_intervals(img_l, img_u, dc, False)
        base_img = base_img.resize((img_width * resize_factor, img_width * resize_factor))
        base_img.show()
        input("press any key to continue")

visualize_cf()


print('done')
