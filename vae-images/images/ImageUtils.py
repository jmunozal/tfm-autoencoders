import glob
import os

from PIL import Image, ImageEnhance
from images.Utils import *
from matplotlib import pyplot as plt

from PIL import Image

def enhance_image(image):
    enhancer = ImageEnhance.Brightness(image)
    img = enhancer.enhance(4)
    return img

def create_rgb_images(folder):
    # iterates from all over the vae-images and creates a RGB image
    # image is enhanced (bright increased)
    for root, dirs, f in os.walk(folder, topdown=False):
        if check_root_dir(root):
            for file in os.listdir(''.join([root, os.sep, "R"])):
                get_rgb_image(root, file, enhance=True, show=False, save=True)

def create_png_images(folder, dest_folder):
    for file in glob.glob(''.join([folder, os.sep, "*.tiff"])):
        #print (file)
        #print (os.path.splitext(file)[0] + '.png')
        png_filename = os.path.basename(os.path.splitext(file)[0] + '.png')
        im = Image.open(file)
        im.save(''.join([dest_folder, os.sep, png_filename]))


def get_rgb_image(file_path, file_name, enhance = False, show = False, save = False):

    image_folder_red = file_path + "/R"
    image_folder_green = file_path + "/G"
    image_folder_blue = file_path + "/B"

    file_red = image_folder_red + os.path.sep + file_name
    file_green = image_folder_green + os.path.sep + file_name
    file_blue = image_folder_blue + os.path.sep + file_name

    try:
        red_image = Image.open(file_red)
        green_image = Image.open(file_green)
        blue_image = Image.open(file_blue)

    except:
        print("some channel does not exist; skipping.")
        return

    merged_image = Image.merge('RGB', (red_image, green_image, blue_image))

    if enhance:
        merged_image = enhance_image(merged_image)

    if show:
        merged_image.show()

    if save:
        rgb_folder = ''.join([file_path, os.sep, RGB_FOLDER])
        create_dir(rgb_folder)
        merged_image.save(''.join([rgb_folder, os.sep, file_name]))

# resize image
def resize(path, new_width = 128, new_height = 128):

    im = Image.open(path)
    width, height = im.size  # Get dimensions

    max_s = max([width, width])
    if max_s > new_width:
        return None

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))

    return im

def show_histogram(pathimage):

    def RED(R): return '#%02x%02x%02x'%(R,0,0)
    def GREEN(G): return '#%02x%02x%02x'%(0,G,0)
    def BLUE(B):return '#%02x%02x%02x'%(0,0,B)
    i=Image.open(pathimage)
    hst=i.histogram()
    l1=hst[0:256]      # indicates Red
    l2=hst[256:512]  # indicated Green
    l3=hst[512:768]   # indicates Blue
    plt.figure(0)
    for i in range(0, 256):
        plt.bar(i, l1[i], color = 'red',alpha=1)
    plt.figure(1)
    for i in range(0, 256):
        plt.bar(i, l2[i], color = 'green',alpha=1)
    plt.figure(2)
    for i in range(0, 256):
        plt.bar(i, l3[i], color = 'blue',alpha=1)
    plt.show()