# moves from dmso_folders.txt file to segmented_dmso folder
import os

from images.Utils import check_root_dir
from images.ImageUtils import resize

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
            if(check_root_dir(root)):
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


#move_files('dmso_folders_ok.txt', '/Volumes/My Passport/PFM/output/segments_dmso')

#create_rgb_images('/Volumes/My Passport/PFM/output/segments_dmso_resized/Week10_40119')
#create_rgb_images('/Volumes/My Passport/PFM/output/segments_dmso_resized/Week10_40111')
#create_rgb_images('/Volumes/My Passport/PFM/output/segments_dmso_resized/Week10_40115')

