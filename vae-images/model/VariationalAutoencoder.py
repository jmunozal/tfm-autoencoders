import random
from glob import glob
from shutil import copyfile

import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, \
    BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator

from images.Utils import choose_random_files
from utils.callbacks import CustomCallback, step_decay_schedule

import numpy as np
import json
import os
import pickle

import matplotlib.pyplot as plt

class VariationalAutoencoder():

    INPUT_DIM = (128, 128, 3)

    def __init__(self
                 , train_mode
                 , image_folder
                 , run_folder
                 , images_to_generate = 5
                 , input_dim = INPUT_DIM
#                 , encoder_conv_filters
#                 , encoder_conv_kernel_size
#                 , encoder_conv_strides
#                 , decoder_conv_t_filters
#                 , decoder_conv_t_kernel_size
#                 , decoder_conv_t_strides
                 , z_dim = 200
                 , use_batch_norm=True
                 , use_dropout=True
                 ):
        self.name = 'variational_autoencoder'

        self.run_folder = run_folder
        self.train_mode = train_mode

        self.input_dim = self.INPUT_DIM
#        self.encoder_conv_filters = encoder_conv_filters
#        self.encoder_conv_kernel_size = encoder_conv_kernel_size
#        self.encoder_conv_strides = encoder_conv_strides
#        self.decoder_conv_t_filters = decoder_conv_t_filters
#        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
#        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.images_to_generate = images_to_generate
        self.image_folder = image_folder

 #       self.n_layers_encoder = len(encoder_conv_filters)
 #       self.n_layers_decoder = len(decoder_conv_t_filters)

        if self.train_mode:

            self.init_dirs()

            imgs = os.sep.join([image_folder, 'training'])
            self.image_files_generate = choose_random_files(folder=imgs, nfiles=self.images_to_generate)
            self.save_original_images()

        self._build()

    def _build(self):
        encoder_input = Input(shape=self.input_dim, name='encoder_input')

        x = encoder_input

        conv_layer_1 = Conv2D(
            filters=32
            , kernel_size=3
            , strides=2
            , padding='same'
            , name='encoder_conv_1'
        )
        x = conv_layer_1(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
            x = Dropout(rate=0.25)(x)

        conv_layer_2 = Conv2D(
            filters=64
            , kernel_size=3
            , strides=2
            , padding='same'
            , name='encoder_conv_2'
        )
        x = conv_layer_2(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
            x = Dropout(rate=0.25)(x)

        conv_layer_3 = Conv2D(
            filters=64
            , kernel_size=3
            , strides=2
            , padding='same'
            , name='encoder_conv_3'
        )
        x = conv_layer_3(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
            x = Dropout(rate=0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]
        x = Flatten()(x)
        #flatten_size = K.int_shape(x)[1:]

        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)

        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)

        ### THE DECODER

        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        conv_t_layer_3b = Conv2DTranspose(
            filters=64
            , kernel_size=3
            , strides=2
            , padding='same'
            , name='decoder_conv_t_3b'
        )
        x = conv_t_layer_3b(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
            x = Dropout(rate=0.25)(x)

        conv_t_layer_2b = Conv2DTranspose(
            filters=64
            , kernel_size=3
            , strides=2
            , padding='same'
            , name='decoder_conv_t_2b'
        )
        x = conv_t_layer_2b(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
            x = Dropout(rate=0.25)(x)

        conv_t_layer_1b = Conv2DTranspose(
            filters=32
            , kernel_size=3
            , strides=2
            , padding='same'
            , name='decoder_conv_t_1b'
        )
        x = conv_t_layer_1b(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
            x = Dropout(rate=0.25)(x)

        conv_t_layer_0b = Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', name='decoder_conv_t_0b')
        x = conv_t_layer_0b(x)
        #decoder_output = Activation('sigmoid')(x)
        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        ### THE FULL VAE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)

    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        ### COMPILATION
        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis=1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return r_loss + kl_loss

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])

    def train_with_generator(self, data_flow, epochs, steps_per_epoch, run_folder, print_every_n_batches=100,
                             initial_epoch=0, lr_decay=1, ):

        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self, self.image_folder, self.image_files_generate)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint_filepath = os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only=True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only=True, verbose=1)

        callbacks_list = [checkpoint1, checkpoint2, lr_sched, custom_callback]

        self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))

        history = self.model.fit_generator(
            data_flow
            , shuffle=True
            , epochs=epochs
            , initial_epoch=initial_epoch
            , callbacks=callbacks_list
            , steps_per_epoch=steps_per_epoch
        )

        self.acuracy_loss(history=history)
        self.print_model(run_folder)

    def train(self, batch_size, epochs, run_folder, directory, print_every_n_batches=100, initial_epoch=0,
              lr_decay=1):

        x_train = keras.preprocessing.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="int",
            class_names=None,
            color_mode="rgb",
            batch_size=32,
            image_size=(128, 128),
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation="bilinear",
            follow_links=False,
        )


        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint_filepath = os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only=True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only=True,
                                      verbose=1)

        callbacks_list = [checkpoint1, checkpoint2, custom_callback, lr_sched]

        self.model.fit(
            x_train
            , x_train
            , batch_size=batch_size
            , shuffle=True
            , epochs=epochs
            , initial_epoch=initial_epoch
            , callbacks=callbacks_list
        )


    def print_model(self, run_folder):

        with open(run_folder + '/viz/model.txt', 'w') as fh:
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
        with open(run_folder + '/viz/encoder.txt', 'w') as fh:
            self.encoder.summary(print_fn=lambda x: fh.write(x + '\n'))
        with open(run_folder + '/viz/decoder.txt', 'w') as fh:
            self.decoder.summary(print_fn=lambda x: fh.write(x + '\n'))

    def save_original_images(self):

        for image in self.image_files_generate:
            im_source = os.sep.join([self.image_folder, 'training', image])
            im_dest = os.sep.join([self.run_folder, 'images', image])
            copyfile(im_source, im_dest)

    def init_dirs(self):

        if not os.path.exists(self.run_folder):
            os.mkdir(self.run_folder)
            os.mkdir(os.path.join(self.run_folder, 'viz'))
            os.mkdir(os.path.join(self.run_folder, 'images'))
            os.mkdir(os.path.join(self.run_folder, 'weights'))
            os.mkdir(os.path.join(self.run_folder, 'plots'))

        with open(os.path.join(self.run_folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim
                , [32, 64, 64]
                , [3, 3, 3]
                , [2, 2, 2]
                , [64, 64, 32, 3]
                , [3, 3, 3, 3]
                , [2, 2, 2, 1]
                , self.z_dim
                , self.use_batch_norm
                , self.use_dropout
                ], f)

    def acuracy_loss(self, history):

        savefolder = os.sep.join([self.run_folder, '/plots'])

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.savefig(os.sep.join([savefolder, 'loss.png']))