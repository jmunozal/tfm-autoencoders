import os
import shutil

import pathlib
from PIL import Image

from images.ImageUtils import resize

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

# checks if a dir has an RGB dir
def check_rgb_dir(dir):
    path = "".join([dir, os.sep, "RGB"])
    is_rgb_dir = os.path.isdir(path)
    return is_rgb_dir, path

# moves files from /segments_dmso to segments_dmso_resized
# also resizes files
# folders_file: file with folders (by drug)
def move_files(folders_file, old_folder_subst):

    dict = {}

    f = open(folders_file)
    for dmso_folder in f.readlines():
        folder = ''.join([old_folder_subst, os.sep, dmso_folder.rstrip()])
        print("analyizng: " + folder)
        for root, dirs, fi in os.walk(folder, topdown=False):
            if dict.get(root, None) == None:
                dict[root] = True
            else:
                continue
            if check_root_dir(root):
                print("root found: " + root)
                for s in ["R", "G", "B"]:
                    file_folder_path = ''.join([root, os.sep, s])
                    copy_folder_path = file_folder_path.replace('segments_dmso', 'segments_dmso_resized')
                    # creates resized dir
                    os.makedirs(copy_folder_path)
                    # resizes and saves the new file (for each channel)
                    for _, _, files in os.walk(file_folder_path, topdown=False):
                        for fi in files:
                            file_2_resize = ''.join([file_folder_path, os.sep, fi])
                            file_resized = file_2_resize.replace('segments_dmso', 'segments_dmso_resized')
                            img_2_save = resize(file_2_resize)
                            if img_2_save is not None:
                                img_2_save.save(file_resized)

#print(check_root_dir("/Volumes/My Passport/PFM/output/segments_dmso_resized/Week1_22123/B02/B02/1"))

def copy_rgb_files(rgb_path, dest_folder):
    image_path = pathlib.Path(rgb_path)

    for rgb_folder, _, _ in os.walk(rgb_path, topdown=False):
        is_rgb_dir, path = check_rgb_dir(rgb_folder)
        if is_rgb_dir:
            copy_folder_path = str(pathlib.Path(path.replace('segments_dmso_resized', 'RGB')).parent)
            #print("creo dir: " + copy_folder_path)
            #print("copy desde: " + path)
            #print("copy hasta: " + copy_folder_path)
            create_dir(copy_folder_path)
            for filename in os.listdir(path):
                ifile = ''.join([rgb_folder, os.sep, 'RGB', os.sep, filename])
                ofile = ''.join([copy_folder_path, os.sep, filename])
                shutil.copy(ifile, ofile)

