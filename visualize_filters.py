import numpy as np
from PIL import Image, ImageDraw

total_images = 1984
img_width = 28
resize_factor = 10
base_path = '../remote/output/RXCSi/output-10-digit/10-digits-39/'
image_file_path = "../RCFC-kb/data/mnist/mnist_validation_all.txt"
# base_path = '../RCFC-kb/cmake-build-debug/output/out11/'
# image_file_path = "../RCFC-kb/data/mnist/mnist_validation_3_8.txt"
# visualization_file_path: str = base_path + 'visualization.txt'
cl_file_path = base_path + '5000000/classifier.txt'
cf_file_path = base_path + '5000000/code_fragment.txt'
filter_file_path = base_path + '5000000/promising_filter.txt'

min_experience = 20
max_error = 10




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
            if img_l[start_x+x*step, (start_y+y*step)] > lb[y*size + x]:
                img_l[start_x+x*step, (start_y+y*step)] = lb[y*size + x]
            if img_u[start_x+x*step, (start_y+y*step)] < ub[y*size + x]:
                img_u[start_x+x*step, (start_y+y*step)] = ub[y*size + x]


def get_pixel_color(img_l, img_u, x, y):
    color = (255, 0, 0)
    if img_l[x, y] == 1 and img_u[x, y] == 0:  # if the pixel interval has not be initialized then its don't care
        return color
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


def match_filter_with_image(fid, image):
    lb, ub, x, y, size, dilated = filter_data[fid]
    img = image.reshape(784,)
    step = 1
    effective_filter_size = size
    if dilated:
        step = 2
        effective_filter_size = size + size -1;
    match_failed = False # flag that controls if the next position to be evaluated when current does not match
    i = y
    j = x
    k = 0
    while k<size and not match_failed:
        l = 0
        while l<size and not match_failed:
            if(img[i*img_width+j + k*step*img_width+l*step] < lb[k*size+l]
               or img[i*img_width+j + k*step*img_width+l*step] > ub[k*size+l]):
                match_failed = True
            l += 1
        k += 1
    if not match_failed:
        return True
    return False


def visualize_image(img_id, rectangle):

    for img_id in range(total_images):
        img_class, img = get_image(img_id)
        base_img = Image.fromarray(img).convert("RGB")
        dc = ImageDraw.Draw(base_img)  # draw context
        base_img_intervals = Image.fromarray(img).convert("RGB")
        dc_intervals = ImageDraw.Draw(base_img_intervals)  # draw context
        # try to match all filters
        img_l = get_blank_image(1)
        img_u = get_blank_image(0)
        filters_drawn = 0
        for fid in filter_data:
            lb, ub, x, y, size, dilated = filter_data[fid]
            if match_filter_with_image(fid, img):
                filters_drawn += 1
                update_bounds(img_l, img_u, lb, ub, x, y, size, dilated)
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
        visualize_intervals(img_l, img_u, dc_intervals)
        base_img_intervals = base_img_intervals.resize((img_width*resize_factor, img_width*resize_factor))
        base_img_intervals.show()
        print("filters drawn: "+str(filters_drawn))
        input("press any key to continue")


visualize_image(-1, True)





print('done')
