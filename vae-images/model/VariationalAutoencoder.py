from glob import glob

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, \
    BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator

from utils.callbacks import CustomCallback, step_decay_schedule

import numpy as np
import json
import os
import pickle



class VariationalAutoencoder():

    INPUT_DIM = (128, 128, 3)

    def __init__(self
                 , input_dim = INPUT_DIM
#                 , encoder_conv_filters
#                 , encoder_conv_kernel_size
#                 , encoder_conv_strides
#                 , decoder_conv_t_filters
#                 , decoder_conv_t_kernel_size
#                 , decoder_conv_t_strides
                 , z_dim = 200
                 , use_batch_norm=False
                 , use_dropout=False
                 ):
        self.name = 'variational_autoencoder'

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

 #       self.n_layers_encoder = len(encoder_conv_filters)
 #       self.n_layers_decoder = len(decoder_conv_t_filters)

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
        x = LeakyReLU()(x)

        conv_layer_2 = Conv2D(
            filters=64
            , kernel_size=3
            , strides=2
            , padding='same'
            , name='encoder_conv_2'
        )

        x = conv_layer_2(x)
        x = LeakyReLU()(x)

#        conv_layer_3 = Conv2D(
#            filters=64
#            , kernel_size=3
#            , strides=2
#            , padding='same'
#            , name='encoder_conv_3'
#        )

#        x = conv_layer_3(x)
#        x = LeakyReLU()(x)

        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)

        flatten_size = K.int_shape(x)[1:]

        x = Dense(1024, name='outpost', activation='relu')(x)

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

        x = Dense(1024, activation= 'relu')(decoder_input)

        x = Dense(flatten_size[0], activation='relu')(x) #hortera

        x = Reshape(shape_before_flattening)(x)

#        conv_t_layer_3b = Conv2DTranspose(
#            filters=64
#            , kernel_size=3
#            , strides=2
#            , padding='same'
#            , name='decoder_conv_t_3b'
#        )

#        x = conv_t_layer_3b(x)
#        x = LeakyReLU()(x)

        conv_t_layer_2b = Conv2DTranspose(
            filters=64
            , kernel_size=3
            , strides=2
            , padding='same'
            , name='decoder_conv_t_2b'
        )

        x = conv_t_layer_2b(x)
        x = LeakyReLU()(x)

        conv_t_layer_1b = Conv2DTranspose(
            filters=32
            , kernel_size=3
            , strides=2
            , padding='same'
            , name='decoder_conv_t_1b'
        )

        x = conv_t_layer_1b(x)
        x = LeakyReLU()(x)

        decoder_output = Conv2DTranspose(
            filters=3, kernel_size=3, strides=1, padding="same") (x),


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

        #custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint_filepath = os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only=True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only=True, verbose=1)

        callbacks_list = [checkpoint1, checkpoint2, lr_sched]

        self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))

        self.model.fit_generator(
            data_flow
            , shuffle=True
            , epochs=epochs
            , initial_epoch=initial_epoch
            , callbacks=callbacks_list
            , steps_per_epoch=steps_per_epoch
        )