import numpy as np
from PIL import Image, ImageDraw

total_images = 1984
img_width = 28
resize_factor = 10
base_path = '../remote/output/RXCSi/output-10-digit/10-digits-39/'
image_file_path = "../RCFC-kb/data/mnist/mnist_validation_all.txt"
visualization_file_path: str = base_path + 'visualization.txt'
cl_file_path = base_path + '5000000/classifier.txt'
cf_file_path = base_path + '5000000/code_fragment.txt'
filter_file_path = base_path + '5000000/filter.txt'

# load classifiers ids and their code fragment ids
cl_cf = {}
f = open(cl_file_path)
line = f.readline()
while line:
    tokens = line.strip().split()
    cl_id = int(tokens[1])
    cl_fitness = float(tokens[9])
    line = f.readline()
    tokens = line.strip().split()
    cf = []
    for token in tokens:
        cf.append(int(token))
    cl_cf[cl_id] = (cl_fitness, cf)
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


def get_image(img_id, denormalize):
    item = img_file[img_id]
    img_class = int(item[-1])
    data = item[:-1]
    # denormalize
    if denormalize:
        data = data * 255
    data = data.reshape(28, 28)
    return img_class, data



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


def get_pixel_color(img_l, img_u, x, y):
    if img_l[x, y] == 1 and img_u[x, y] == 0:  # if the pixel interval has not be initialized then its don't care
        return "#00008B"
    if img_l[x, y] == 0 and img_u[x, y] == 1:  # if the pixel interval has max then its don't care
        return "#00008B"  #"#006400"
    if img_l[x, y] == 0:  # this interval accepts black
        return "#000000"
    if img_u[x, y] == 1:  # this interval accepts white
        return "#ffffff"
    # if img_u[x, y] - img_l[x, y] > 0.5:  # wide interval means don't care
    #     return color
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


def visualize_intervals(img_l, img_u, dc):
    for y in range(img_width):
        for x in range(img_width):
            dc.point((x, y), get_pixel_color(img_l, img_u, x,y))


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
        cl_ids = []
        for id in tokens:
            cl_ids.append(int(id))
        visualization_data[read_img_id] = (actual_class, predicted_class, cl_ids)
        line = f.readline()  # classifier_id predicted_class ...
    f.close()


load_visualization_data()

def match_filter_with_image(fid, image):
    lb, ub, x, y, size, dilated = filter_data[fid]

    img = image.reshape(784,)
    step = 1
    effective_filter_size = size
    if dilated:
        step = 2
        effective_filter_size = size + size -1;
    match_failed = False  # flag that controls if the next position to be evaluated when current does not match
    iy = y
    ix = x
    fy = 0
    while fy < size and not match_failed:
        fx = 0
        while fx < size and not match_failed:
            if(img[iy*img_width+ix + fy*step*img_width+fx*step] < lb[fy*size+fx]
               or img[iy*img_width+ix + fy*step*img_width+fx*step] > ub[fy*size+fx]):
                match_failed = True
            fx += 1
        fy += 1
    if not match_failed:
        return True
    return False


def filter_color(lb, ub, size, matched):
    if sum(lb) == 0 and sum(ub) == size * size:
        assert matched
        return 0  # don't care
    if matched:
        if sum(lb) == 0:
            return "#000000"
        elif sum(ub) == size*size:
            return "#ffffff"
        else:
            return "#696969"
    else:
        if sum(lb) == 0:
            return "#ffffff"
        elif sum(ub) == size*size:
            return "#000000"
        else:
            return 0  #"#696969"  #"#ff0000"


def invert_bounds(lb, ub, size):
    if sum(lb) == 0 and sum(ub) == size*size:
        assert False
    lb1 = 0
    ub1 = 0
    lb2 = ub2 = 0
    if sum(lb) == 0:
        lb1 = ub.copy()
        ub1 = ub.copy()
        for i in range(size*size):
            ub1[i] = 1
    elif sum(ub) == size*size:
        ub1 = lb.copy()
        lb1 = lb.copy()
        for i in range(size*size):
            lb1[i] = 0
    # else:
    #     lb1 = lb.copy()
    #     for i in range(size*size):
    #         lb1[i] = 0
    #     ub1 = lb.copy()
    #
    #     lb2 = ub.copy()
    #     ub2 = ub.copy()
    #     for i in range(size*size):
    #         ub2[i] = 1
    return lb1, ub1, lb2, ub2

def visualize_image(img_id, rectangle, visualize_wrongly_classified):

    for img_id in range(total_images):
        img_class, img = get_image(img_id, True)
        original_image = Image.fromarray(img).convert("RGB")
        # base_img = Image.fromarray(get_blank_image(220)).convert("RGB")
        base_img = Image.new("RGB", (img_width, img_width), "#00008B")
        dc = ImageDraw.Draw(base_img)  # draw context
        base_img_intervals = Image.fromarray(img).convert("RGB")
        dc_intervals = ImageDraw.Draw(base_img_intervals)  # draw context
        actual_class, predicted_class, cl_ids = visualization_data[img_id]
        if visualize_wrongly_classified and actual_class == predicted_class:
            continue
        print("image id: " + str(img_id) + " actual class: " + str(actual_class) + " predicted class: " + str(predicted_class))
        img_l = get_blank_image(0)
        img_u = get_blank_image(1)
        filters_drawn = 0
        img = img / 255  # normalize again for filter matching
        for cl_id in cl_ids:
            already_processed_filters = {}
            # get classifier code fragments
            cl_fitness, code_fragments = cl_cf[cl_id]
            # if cl_fitness < .1:
            #     continue
            for cf in code_fragments:
                if cf == -1:
                    continue
                filters = cf_filter[cf]  # filter
                for filter in filters:
                    if filter in already_processed_filters:
                        continue
                    already_processed_filters[filter] = 1
                    lb, ub, x, y, size, dilated = filter_data[filter]
                    # if dilated:
                    #     continue
                    filters_drawn += 1
                    matched = match_filter_with_image(filter, img)
                    if matched:
                        update_bounds(img_l, img_u, lb, ub, x, y, size, dilated)
                    else:
                        lb1, ub1, lb2, ub2 = invert_bounds(lb, ub, size)
                        if lb1 != 0 and ub1 != 0:
                            update_bounds(img_l, img_u, lb1, ub1, x, y, size, dilated)
                        if lb2 != 0 and ub2 != 0:
                            update_bounds(img_l, img_u, lb2, ub2, x, y, size, dilated)
                    if dilated:
                        size = size * 2 - 1
                    fill = filter_color(lb, ub, size, matched)
                    if fill != 0:
                        if rectangle:
                            shape = [(x, y), (x + size-1, y + size-1)]
                            dc.rectangle(shape, fill=fill)
                        else:
                            # center point
                            x += size // 2
                            y += size // 2
                            dc.point((x, y), fill=fill)
                    stop = 0


        base_img = base_img.resize((img_width*resize_factor, img_width*resize_factor))
        base_img.show()
        visualize_intervals(img_l, img_u, dc_intervals)
        base_img_intervals = base_img_intervals.resize((img_width*resize_factor, img_width*resize_factor))
        base_img_intervals.show()
        original_image = original_image.resize((img_width*resize_factor, img_width*resize_factor))
        original_image.show()
        print("filters drawn: "+str(filters_drawn))
        input("press any key to continue")


visualize_image(-1, True, False)





print('done')
