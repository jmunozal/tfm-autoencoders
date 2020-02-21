from PIL import Image, ImageEnhance
import os

def enhance_image(image):
    enhancer = ImageEnhance.Brightness(image)
    img = enhancer.enhance(4)
    return img


def get_rgb_image(file_path, file_name, enhance = False, show = False, save = False):
    image_folder_red = file_path + "/R"
    image_folder_green = file_path + "/G"
    image_folder_blue = file_path + "/B"

    file_red = image_folder_red + os.path.sep + file
    file_green = image_folder_green + os.path.sep + file
    file_blue = image_folder_blue + os.path.sep + file

    try:
        red_image = Image.open(file_red)
        green_image = Image.open(file_green)
        blue_image = Image.open(file_blue)
    except:
        print("some channel does not exist; skipping.")
        return

#    if red_image.size[0] > limit or red_image.size[1] > limit:
#        print("image " + itera + " too large; skipping.")
#        return

    merged_image = Image.merge('RGB', (red_image, green_image, blue_image))

    if enhance:
        merged_image = enhance_image(merged_image)

    if show:
        merged_image.show()

    if save:
        merged_image.save(file_path + os.path.sep + file_name)


file = "CELLS_25.tiff"
path = "/Volumes/My Passport/PFM/output/0_Week1_150607_None_0_1_None_B02"
get_rgb_image(file_name=file, file_path=path, enhance=True, show=True)