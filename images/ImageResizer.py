from os import path

from PIL import Image

from Utils import SEGMENTS_PATH
import os
import matplotlib.pyplot as pl
import numpy as np

BASE_DIR = "R"
NEW_WIDTH, NEW_HEIGHT = 128, 128

# resize image
def resize(path):

    im = Image.open(path)
    width, height = im.size  # Get dimensions

    max_s = max([width, width])
    if max_s > NEW_WIDTH:
        return None

    left = (width - NEW_WIDTH) / 2
    top = (height - NEW_HEIGHT) / 2
    right = (width + NEW_WIDTH) / 2
    bottom = (height + NEW_HEIGHT) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))

    return im


# iterates images
for root, dirs, f in os.walk(SEGMENTS_PATH, topdown=False):
    if (root.endswith(BASE_DIR)):
        R_DIR = root
        G_DIR = root[:-1] + "G"
        B_DIR = root[:-1] + "B"

        R_DIR_RESIZED = R_DIR.replace("segments_dmso", "segments_dmso_resized")
        G_DIR_RESIZED = G_DIR.replace("segments_dmso", "segments_dmso_resized")
        B_DIR_RESIZED = B_DIR.replace("segments_dmso", "segments_dmso_resized")

        if not os.path.isdir(R_DIR_RESIZED):
            os.makedirs(R_DIR_RESIZED)
            os.makedirs(G_DIR_RESIZED)
            os.makedirs(B_DIR_RESIZED)

        for file in f:

            resized_r = resize(R_DIR + "/" + file)
            resized_g = resize(G_DIR + "/" + file)
            resized_b = resize(B_DIR + "/" + file)

            if resized_r is None:
                continue

            resized_r.save(R_DIR_RESIZED + "/" + file)
            resized_g.save(G_DIR_RESIZED + "/" + file)
            resized_b.save(B_DIR_RESIZED + "/" + file)


