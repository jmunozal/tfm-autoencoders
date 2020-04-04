# moves from dmso_folders.txt file to segmented_dmso folder
import os

from images.Utils import check_root_dir, move_files, copy_rgb_files
from images.ImageUtils import resize, print_image_data

#move_files('dmso_folders_ok.txt', '/Volumes/My Passport/PFM/output/segments_dmso')

#create_rgb_images('/Volumes/My Passport/PFM/output/segments_dmso_resized/Week10_40119')
#create_rgb_images('/Volumes/My Passport/PFM/output/segments_dmso_resized/Week10_40111')
#create_rgb_images('/Volumes/My Passport/PFM/output/segments_dmso_resized/Week10_40115')

#copy_rgb_files('/Volumes/My Passport/PFM/output/segments_dmso_resized', '/Volumes/My Passport/PFM/output/RGB')

print_image_data('/Volumes/My Passport/PFM/output/RGB/Week1_22123_B02_1_Cells_156.tiff')