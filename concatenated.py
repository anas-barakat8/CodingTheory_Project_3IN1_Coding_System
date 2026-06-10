# this is the concateneted code the contains both the hamming code and the convolutional code
# since there is no need to implemet it as a separate class, I will just call the functions of the hamming code and the convolutional code.
# the main idea is to use Hamming encoder then convolutional encoder and then use the convolutional decoder and then the hamming decoder.


import hamming
import convolutional
import numpy as np
import parameters as params 
from utilities import *


def concatenated_encode(message_bits):
    
    hamming_code = hamming.HammingCode(params.H, params.k)
    convolutional_code = convolutional.ConvolutionalCode(params.generators, params.K, params.memory)


    hamming_encoded_bits = hamming_code.encode(message_bits)

    concatenated_encoded_bits = convolutional_code.encode( hamming_encoded_bits,terminate=True )

    return concatenated_encoded_bits


