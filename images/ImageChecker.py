from os import path

from PIL import Image

from Utils import SEGMENTS_PATH
import os

BASE_DIR = "R/"

# check all images are coherent
for root, dirs, f in os.walk(SEGMENTS_PATH, topdown=False):

    # folder walk
    for name in dirs:
        if name == BASE_DIR:
            R_DIR = os.path.join(root, BASE_DIR) # R folder
            G_DIR = os.path.join(root, "G/")  # G folder
            B_DIR = os.path.join(root, "B/")  # B folder
            #print(R_DIR)
            #print(G_DIR)
            #print(B_DIR)

            for _, _, files in os.walk(R_DIR, topdown=False):
                for filename in files:
                    # check if other channels exist
                    if path.exists(G_DIR + filename) == False or path.exists(B_DIR + filename) == False:
                        print("Complementary file of: " + R_DIR + "does not exist")
                        continue

                    red_image = Image.open(R_DIR + filename)
                    green_image = Image.open(G_DIR + filename)
                    blue_image = Image.open(B_DIR + filename)

                    width_r, height_r = red_image.size
                    width_g, height_g = green_image.size
                    width_b, height_b = blue_image.size

                    if width_r != width_g != width_b:
                        print ("width: size not equal in " + R_DIR + filename)

                    if height_r != height_g != height_b:
                        print ("height: size not equal in " + R_DIR + filename)

