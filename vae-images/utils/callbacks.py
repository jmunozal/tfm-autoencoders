from PIL import Image
from keras.callbacks import Callback, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import os

#### CALLBACKS
from keras_preprocessing.image import img_to_array, array_to_img


class CustomCallback(Callback):
    
    def __init__(self, run_folder, print_every_n_batches, initial_epoch, vae, images_folder, images_files):
        self.epoch = initial_epoch
        self.run_folder = run_folder
        self.print_every_n_batches = print_every_n_batches
        self.vae = vae
        self.images_folder = images_folder
        self.images_files = images_files

        print(*images_files)

#    def on_batch_end(self, batch, logs={}):
#        if batch % self.print_every_n_batches == 0:
#            z_new = np.random.normal(size = (1,self.vae.z_dim))
#            reconst = self.vae.decoder.predict(np.array(z_new))[0].squeeze()#
#
#            filepath = os.path.join(self.run_folder, 'images', 'img_' + str(self.epoch).zfill(3) + '_' + str(batch) + '.jpg')
#            if len(reconst.shape) == 2:
#                plt.imsave(filepath, reconst, cmap='gray_r')
#            else:
#                plt.imsave(filepath, reconst)


    def on_epoch_end(self, epoch, logs=None):
        self.predict_samples(epoch=epoch)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1

    def predict_samples(self, epoch):
        suffix = str(self.epoch).zfill(3) + '_' + str(epoch)
        for i in self.images_files:
            print(i)
            filename = i.split('.')[0]
            filename = '_'.join([filename, suffix])
            print(os.sep.join([self.images_folder, i]))
            img = Image.open(os.sep.join([self.images_folder, 'training', i]))
            array_img = img_to_array(img)  # keras
            array_img = np.expand_dims(array_img, axis=0)
            z_result = self.vae.encoder.predict(array_img)
            reco = self.vae.decoder.predict(z_result).squeeze()
            outimg = array_to_img(reco, scale=True)
            outimg.save(os.sep.join([self.run_folder, 'images', filename + '.png']))


def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch/step_size))
        
        return new_lr

    return LearningRateScheduler(schedule)