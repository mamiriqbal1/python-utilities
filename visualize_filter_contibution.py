import numpy as np
from PIL import Image, ImageDraw

total_images = 1984
img_width = 28
resize_factor = 10
base_path = '../XCS-IMG/cmake-build-debug/output-10-digit/10-digits-11/'
image_file_path = "../XCS-IMG/data/mnist/mnist_train_all.txt"
visualization_file_path: str = base_path + 'visualization_train.txt'
filter_file_path = base_path + '3350000/filter.txt'


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


def get_pixel_color(img_l, img_u, x, y, lower):
    if img_l[x, y] == -1 and img_u[x, y] == 2:  # if the pixel interval has not be initialized then its don't care
        return "#000000"  # "#7DCEA0"  # "#0000ff"
    if img_l[x, y] == 0 and img_u[x, y] == 1:  # if the pixel interval has max then its don't care
        return "#000000"  # "#7DCEA0"  #ff0000"  #"#006400"
    if img_l[x, y] == 0:  # this interval accepts black
        return "#000000"
    if img_u[x, y] == 1:  # this interval accepts white
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

    mid = (img_l[x,y] + img_u[x,y]) / 2
    # real to 255 scale
    c = int(mid*255)
    color = (c, c, c)

    # color = "ff0000"
    # if mid < 0.25:
    #     color = "000000"  # black
    # elif mid < 0.5:
    #     color = "D3D3D3"  # light grey

    return color


def visualize_intervals(img_l, img_u, dc, lower):
    for y in range(img_width):
        for x in range(img_width):
            dc.point((x, y), get_pixel_color(img_l, img_u, x,y, lower))


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
        all_filters = []
        for id in tokens:
            all_filters.append((int(id), -1))
        line = f.readline()  # classifier_id predicted_class ...
        tokens = line.strip().split()
        # every id is followed by location where filter did not match
        is_id = True
        id = -1
        location = -1
        for item in tokens:
            if is_id:
                id = int(item)
                is_id = False
            else:
                location = int(item)
                is_id = True
                all_filters.append((id, location))
        visualization_data[read_img_id] = (actual_class, predicted_class, all_filters)
        line = f.readline()  # classifier_id predicted_class ...
    f.close()


load_visualization_data()


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


# invert single interval
def invert_bounds_location(lb, ub):
    if lb == 0:
        lb = ub
        ub = 1
    elif ub == 1:
        ub = lb
        lb = 0
    else:  # lb > 0 and ub < 1:
        # make it "not initialized"
        lb = -1
        ub = 2
    return lb, ub

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


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def visualize_image(img_id_only, rectangle, visualize_wrongly_classified, digit):

    for img_id in range(total_images):
        if img_id_only != -1:
            img_id = img_id_only
        img_class, img = get_image(img_id, True)
        original_image = Image.fromarray(img).convert("RGB")
        # base_img = Image.fromarray(get_blank_image(220)).convert("RGB")
        base_img = Image.new("RGB", (img_width, img_width), "#00008B")
        dc = ImageDraw.Draw(base_img)  # draw context
        base_img_intervals_lower = Image.fromarray(img).convert("RGB")
        dc_intervals_lower = ImageDraw.Draw(base_img_intervals_lower)  # draw context
        base_img_intervals_upper = Image.fromarray(img).convert("RGB")
        dc_intervals_upper = ImageDraw.Draw(base_img_intervals_upper)  # draw context
        actual_class, predicted_class, all_filters = visualization_data[img_id]
        if digit != -1 and actual_class != digit:
            continue
        if visualize_wrongly_classified and actual_class == predicted_class:
            continue
        print("image id: " + str(img_id) + " actual class: " + str(actual_class) + " predicted class: " + str(predicted_class))
        # initialize bounds to see if they are updated lower = 1, upper = 0
        img_l = get_blank_image(-1)
        img_u = get_blank_image(2)
        filters_drawn = 0
        img = img / 255  # normalize again for filter matching

        already_processed_filters = {}
        for filter_pair in all_filters:
            filter = filter_pair[0]
            location = filter_pair[1]
            # if location == -1:  # process only negative filters
            #     continue
            if filter in already_processed_filters:
                continue
            already_processed_filters[filter] = 1
            lbf, ubf, x, y, size, dilated = filter_data[filter]
            lb = lbf.copy()
            ub = ubf.copy()
            # if dilated:
            #     continue
            filters_drawn += 1
            matched = location == -1
            if matched:
                update_bounds(img_l, img_u, lb, ub, x, y, size, dilated)
            else:
                # since matched failed due to only one location, just invert single interval
                lb_location, ub_location = invert_bounds_location(lb[location], ub[location])
                # now update the filter lb and ub accordingly
                # all the intervals before the location stays the same
                # interval a location is inverted
                lb[location] = lb_location
                ub[location] = ub_location
                # all the intervals after the location are don't care
                for i in range(location+1, size*size):
                    lb[i] = -1
                    ub[i] = 2
                update_bounds(img_l, img_u, lb, ub, x, y, size, dilated)
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


        # base_img = base_img.resize((img_width*resize_factor, img_width*resize_factor))
        # base_img.show()
        original_image = original_image.resize((img_width*resize_factor, img_width*resize_factor))
        # original_image.show()
        visualize_intervals(img_l, img_u, dc_intervals_lower, True)
        base_img_intervals_lower = base_img_intervals_lower.resize((img_width*resize_factor, img_width*resize_factor))
        # base_img_intervals_lower.show()
        visualize_intervals(img_l, img_u, dc_intervals_upper, False)
        base_img_intervals_upper = base_img_intervals_upper.resize((img_width*resize_factor, img_width*resize_factor))
        # base_img_intervals_upper.show()
        all_img = get_concat_h(original_image, base_img_intervals_lower)
        all_img = get_concat_h(all_img, base_img_intervals_upper)
        all_img.show()
        if img_id_only != -1:
            exit(0)
        print("filters drawn: "+str(filters_drawn))
        input("press any key to continue")


visualize_image(-1, True, True, -1)





print('done')
