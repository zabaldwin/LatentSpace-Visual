import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Flatten
from tensorflow.keras.models import Model
import numpy as np                                                                                                                                                                                    
import pandas as pd
from qkeras import QActivation, QConv2D, QDense, quantized_bits
        
import subprocess
import sys  


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Need to downgrade TensorFlow to 2.9 for compatibility with .tf weights
install("tensorflow==2.12")
install("qkeras")  

# Custom layers defined in the original Convolutional Autoencoder (CAE)
class KerasPaddingLayer(tf.keras.layers.Layer):
    def call(self, x):
        padding = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]]) #pad height and width with 1 row/column.
        return tf.pad(x, padding, mode='CONSTANT', constant_values=0)

    def compute_output_shape(self, input_shape):
        batch_size, height, width, channels = input_shape #increase height and width by 1 -- keeps other dimensions the same
        return (batch_size, height + 1, width + 1, channels)


class KerasMinimumLayer(tf.keras.layers.Layer):
    def __init__(self, saturation_value=1, **kwargs):
        super(KerasMinimumLayer, self).__init__(**kwargs)
        self.saturation_value = saturation_value

    def call(self, x):
        return tf.minimum(x, self.saturation_value) #cap values at saturation_value

    def compute_output_shape(self, input_shape):
        return input_shape


class KerasFloorLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.math.floor(x) #round down each element

    def compute_output_shape(self, input_shape):
        return input_shape
     

# Encoder model setup
class EncoderModelBuilder:
    @staticmethod
    def build_encoder_model():
        input_shape = (8, 8, 1) #shape of the wafer input
        condition_shape = (8,) #shape of the condition input

        wafer_input = Input(shape=input_shape, name='Wafer_Input')
        condition_input = Input(shape=condition_shape, name='Condition_Input')

        x = QActivation(activation=quantized_bits(bits=8, integer=1), name='Input_Quantization')(wafer_input)
        x = KerasPaddingLayer()(x)
        x = QConv2D(
            filters=8, kernel_size=3, strides=2, padding='valid',
            kernel_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
            bias_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
            name='Conv2D'
        )(x)
        x = QActivation(activation=quantized_bits(bits=8, integer=1), name='Activation')(x)
        x = Flatten()(x)
        x = QDense(
            units=16,
            kernel_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
            bias_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
            name='Dense_Layer'
        )(x)
        x = QActivation(activation=quantized_bits(bits=9, integer=1), name='Latent_Quantization')(x)
        latent_output = x

        # Optional quantization steps -- this can further refine the latent vector
        bits_per_output = 9
        if bits_per_output > 0:
            n_integer_bits = 1
            n_decimal_bits = bits_per_output - n_integer_bits
            output_max_int_size = 1 << n_decimal_bits
            output_saturation_value = (1 << n_integer_bits) - 1. / (1 << n_decimal_bits)

            latent_output = KerasFloorLayer()(latent_output * output_max_int_size)
            latent_output = KerasMinimumLayer(saturation_value=output_saturation_value)(latent_output / output_max_int_size)

        latent_output = Concatenate(axis=1)([latent_output, condition_input])

        encoder = Model(inputs=[wafer_input, condition_input], outputs=latent_output, name='Encoder_Model')
        return encoder

def convert_weights(tf_weights_path, h5_weights_path):
    
    encoder = EncoderModelBuilder.build_encoder_model()
    encoder.summary()

    # Load the .tf weights
    try:
        encoder.load_weights(tf_weights_path)
        print(f"Successfully loaded weights from {tf_weights_path}.")
    except ValueError as e:
        print(f"Error loading weights: {e}")
        return

    # Save the weights in updated .h5 format
    encoder.save_weights(h5_weights_path)
    print(f"Weights successfully converted to HDF5 format and saved as '{h5_weights_path}'.")

# Replace 'best-encoder-epoch.tf' with the path to the .tf weights
# should keep updated converted model eLinks in the same directory
if __name__ == "__main__":
    convert_weights('./model_5_eLinks/best-encoder-epoch.tf', './model_5_eLinks/best-encoder-epoch-converted.h5')
