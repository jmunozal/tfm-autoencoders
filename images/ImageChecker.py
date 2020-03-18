from os import path
from PIL import Image
from Utils import SEGMENTS_PATH
import os
import matplotlib.pyplot as pl
import numpy as np
import argparse

BASE_DIR = "R"

M = np.array([])
mis = 0;

# check all images are coherent
for root, dirs, f in os.walk(SEGMENTS_PATH, topdown=False):
    #print("root: " + root)

    # folder walk
    for name in dirs:
        #print ("name: " + name)
        if name == BASE_DIR:
            R_DIR = os.path.join(root, BASE_DIR + "/") # R folder
            G_DIR = os.path.join(root, "G/")  # G folder
            B_DIR = os.path.join(root, "B/")  # B folder
            #print(R_DIR)
            #print(G_DIR)
            #print(B_DIR)
            print("R_dir: " + R_DIR)

            for _, _, files in os.walk(R_DIR, topdown=False):
                for filename in files:
                    #print("filename: " + filename)

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

                    max_s = max([width_r, height_r])
                    if max_s < 250:
                        M = np.append(M, max_s)
                        mis = max([mis, max([width_r, height_r])])


print("Total number of images: " + str(len(M)))
print("Max image size: " + str(mis))
print("Image average size: " + str(np.average(M)))
fig = pl.hist(M, bins=500, density=0)
pl.title('Size Histogram')
pl.xlabel("max image width/height")
pl.ylabel("n")
pl.savefig("histogram.png")