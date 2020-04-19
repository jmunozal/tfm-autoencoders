import os
from pathlib import Path

from PIL import Image
from keras_preprocessing.image import img_to_array, np, array_to_img

from images.Utils import choose_random_files
from model.VariationalAutoencoder import VariationalAutoencoder

IMAGES_FOLDER = '/Volumes/My Passport/PFM/output/training_png'
MODEL_FOLDER = os.environ.get('HOME') + '/model3'
DATA_FOLDER = '/Volumes/My Passport/PFM/fastcheck'
RUN_FOLDER = '/Volumes/My Passport/PFM/run/'

OUT_FOLDER = '/Users/kepler/Google Drive/PFM/PEC2/images'

images = choose_random_files(IMAGES_FOLDER, 20)

vae = VariationalAutoencoder(train_mode=False, run_folder= None, image_folder=None)
vae.model.load_weights(os.path.join(MODEL_FOLDER, 'weights/weights.h5'))

for image in images:

    image_path = os.sep.join([IMAGES_FOLDER, image])

    filename = Path(image_path).stem

    img = Image.open(image_path)

    array_img = img_to_array(img)
    array_img = np.expand_dims(array_img, axis=0)

    z_result = vae.encoder.predict(array_img)
    reco = vae.decoder.predict(z_result).squeeze()

    outimg = array_to_img(reco, scale=True)

    img.save(os.sep.join([OUT_FOLDER, image]))
    outimg.save(os.sep.join([OUT_FOLDER, filename + "_RECO.png"]))


