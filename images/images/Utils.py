import os
from PIL import Image

SEGMENTS_PATH = "/Volumes/My Passport/PFM/output/segments_dmso/"
SEGMENTS_RESIZED_PATH = "/Volumes/My Passport/PFM/output/segments_dmso_resized/"

BASE_DIR = "R"
NEW_WIDTH, NEW_HEIGHT = 128, 128
RGB_FOLDER = "RGB"

def create_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

# checks if a dir has R G B channels (is a root dir)
def check_root_dir(dir):
    is_dir = True
    for s in ["R", "G", "B"]:
        is_dir = is_dir and os.path.isdir("".join([dir, os.sep, s]))
    return is_dir

#print(check_root_dir("/Volumes/My Passport/PFM/output/segments_dmso_resized/Week1_22123/B02/B02/1"))
