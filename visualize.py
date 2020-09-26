import numpy as np
from PIL import Image, ImageDraw

total_images = 1984
img_width = 28
resize_factor = 10
base_path = '../remote/output/RXCSi/output-2-digit/2-digits-21/'
image_file_path = "../RCFC-kb/data/mnist/mnist_validation_3_8.txt"
visualization_file_path: str = base_path + 'visualization.txt'
cl_file_path = base_path + '999800/classifier.txt'
cf_file_path = base_path + '999800/code_fragment.txt'
filter_file_path = base_path + '999800/filter.txt'

# load classifiers ids and their code fragment ids
cl_cf = {}
f = open(cl_file_path)
line = f.readline()
while line:
    tokens = line.strip().split()
    cl_id = int(tokens[1])
    line = f.readline()
    tokens = line.strip().split()
    cf = []
    for token in tokens:
        cf.append(int(token))
    cl_cf[cl_id] = cf
    line = f.readline()
f.close()

# load code fragments and their filter ids
cf_filter = {}
f = open(cf_file_path)
line = f.readline()
while line:
    tokens = line.strip().split()
    cl_id = int(tokens[0])
    filters = []
    for token in tokens:
        if token.startswith("D"):
            filters.append(int(token[1:]))
    cf_filter[cl_id] = filters
    line = f.readline()
f.close()

img_file = np.loadtxt(image_file_path)


def get_blank_image(val):
    data = np.zeros((img_width, img_width))
    data += val
    return data

def get_image(img_id):
    item = img_file[img_id]
    img_class = int(item[-1])
    data = item[:-1]
    # denormalize
    data = data * 255
    data = data.reshape(28, 28)
    return img_class, data

img = get_image(0)
img_l = get_blank_image(1)
img_u = get_blank_image(0)

filter_data = {}

def load_filter_data():
    f_id = -1
    x = -1
    y = -1
    size = -1
    dilated = False
    f = open(filter_file_path)
    line = f.readline()
    while line:
        tokens = line.strip().split()
        f_id = int(tokens[1])
        x = int(tokens[3])
        y = int(tokens[5])
        size = int(tokens[7])
        dilated = bool(int(tokens[9]))
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
        filter_data[f_id] = (lb, ub, x, y, size, dilated)


load_filter_data()


# update lower and upper bounds from filter
def update_bounds(lb, ub, start_x, start_y, size, dilated):
    step = 1
    if dilated:
        step = 2
        effective_size = size + size - 1

    for y in range(size):
        for x in range(size):
            if img_l[start_x+x*step, (start_y+y*step)] > lb[y*size + x]:
                img_l[start_x+x*step, (start_y+y*step)] = lb[y*size + x]
            if img_u[start_x+x*step, (start_y+y*step)] < ub[y*size + x]:
                img_u[start_x+x*step, (start_y+y*step)] = ub[y*size + x]


def get_pixel_color(x,y):
    color = (255, 0, 0)
    if img_l[x, y] == 1 and img_u[x, y] == 0:  # if the pixel interval has not be initialized then its don't care
        return color
    # if img_l[x, y] < 0.25 and img_u[x, y] > 0.75:  # wide interval means don't care
    #     return color

    mid = (img_l[x,y] + img_u[x,y]) / 2
    # real to 255 scal
    # e
    c = int(mid*255)
    color = (c, c, c)

    # color = "ff0000"
    # if mid < 0.25:
    #     color = "000000"  # black
    # elif mid < 0.5:
    #     color = "D3D3D3"  # light grey

    return color


def visualize_intervals(dc):
    for y in range(img_width):
        for x in range(img_width):
            dc.point((x, y), get_pixel_color(x,y))


visualization_data = {}


def load_visualization_data():
    # load visualization data
    actual_class = -1
    predicted_class = -1
    f = open(visualization_file_path)
    line = f.readline()
    while line:
        cl_clclass = []
        tokens = line.strip().split()
        read_img_id = int(tokens[0])
        actual_class = int(tokens[1])
        predicted_class = int(tokens[2])
        # print("image id: " + str(read_img_id) + " actual class: " + str(actual_class) + " predicted class: " + str(predicted_class))
        line = f.readline()  # classifier_id predicted_class ...
        tokens = line.strip().split()
        i = 0
        while i < len(tokens):
            classifier = tokens[i]
            i += 1
            classifier_class = tokens[i]
            i += 1
            cl_clclass.append((int(classifier), int(classifier_class)))
        visualization_data[read_img_id] = (actual_class, predicted_class, cl_clclass)
        line = f.readline()  # classifier_id predicted_class ...
    f.close()


load_visualization_data()


def visualize_image(img_id, rectangle, visualize_wrongly_classified):

    for img_id in range(total_images):
        img_class, img = get_image(img_id)
        base_img = Image.fromarray(img).convert("RGB")
        dc = ImageDraw.Draw(base_img)  # draw context
        base_img_intervals = Image.fromarray(img).convert("RGB")
        dc_intervals = ImageDraw.Draw(base_img_intervals)  # draw context
        actual_class, predicted_class, cl_clclass = visualization_data[img_id]
        if visualize_wrongly_classified and actual_class == predicted_class:
            continue
        print("image id: " + str(img_id) + " actual class: " + str(actual_class) + " predicted class: " + str(predicted_class))
        filters_drawn = 0
        for classifier in cl_clclass:
            if classifier[1] == predicted_class:
                classifier_id = classifier[0]
                # get classifier code fragments
                code_fragments = cl_cf[classifier_id]
                for cf in code_fragments:
                    filters = cf_filter[cf]  # filter
                    for filter in filters:
                        filters_drawn += 1
                        lb, ub, x, y, size, dilated = filter_data[filter]
                        update_bounds(lb, ub, x, y, size, dilated)
                        if rectangle:
                            shape = [(x, y), (x + size, y + size)]
                            dc.rectangle(shape, fill="#0000ff")
                        else:
                            # center point
                            x += size // 2
                            y += size // 2
                            dc.point((x, y), fill="#0000ff")

        base_img = base_img.resize((img_width*resize_factor, img_width*resize_factor))
        base_img.show()
        visualize_intervals(dc_intervals)
        base_img_intervals = base_img_intervals.resize((img_width*resize_factor, img_width*resize_factor))
        base_img_intervals.show()
        print("filters drawn: "+str(filters_drawn))
        input("press any key to continue")


visualize_image(-1, True, True)





print('done')
