import os

from PIL import Image
from keras_preprocessing.image import img_to_array, np, array_to_img

def calculate_metrics(vae, path):

    images = 0
    diff = 0;
    for file in os.listdir(path):

        image_ = os.sep.join([path, file])
        image = Image.open(image_)
        predicted = predict(image= image, vae=vae)
        diff += get_diff(image, predicted)
        images += images

    set_val = diff / images

    print('Total number of images: ' + images)
    print('Value of the metrics: ' + images)

def predict(image, vae):

    array_img = img_to_array(image)
    array_img = np.expand_dims(array_img, axis=0)
    z_result = vae.encoder.predict(array_img)
    reco = vae.decoder.predict(z_result).squeeze()
    outimg = array_to_img(reco, scale=True)
    return outimg


def get_diff(image1, image2):

    array_img_1 = img_to_array(image1)
    array_img_2 = img_to_array(image2)
    df = np.asarray(array_img_1 - array_img_2)
    dst = np.sum(abs(df), axis=1)
    return sum(sum(dst) / (128. * 128. * 256.)) / 3. # for each channel

#a = np.zeros((128, 128, 3))
#b = np.zeros((128, 128, 3))
#b.fill(256.)
#imga = array_to_img(a)
#imgb = array_to_img(a)
#imga.show()
#imgb.show()
#print(get_diff(imgb, imga))
#img1 = Image.open(image_path)
#img2 = Image.open(image_path)