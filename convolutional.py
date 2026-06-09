import numpy as np
from utilities import *


class ConvolutionalCode:
    # left state out of member variables for easier handling and debugging
    def __init__(self, generators, constraint_length, memory_length):
        self.generators = np.array(generators, dtype=np.uint8)
        self.constraint_length = constraint_length
        self.memory = memory_length
        self.num_states = 2**self.memory

    def output_bits(self, input_bit, state):# this calculate the output bits based on the generator polynomials and the register state
        state_bits = self.state_to_bits(state)
        register = np.concatenate(([input_bit], state_bits))

        output = []

        for generator in self.generators:
            selected_bits = register * generator
            output_bit = np.mod(np.sum(selected_bits), 2)
            output.append(output_bit)

        return np.array(output, dtype=np.uint8)

    def next_state(self, input_bit, state): # calculate the next state based on the current state and the input bit
        state_bits = self.state_to_bits(state)

        new_state_bits = np.concatenate(([input_bit], state_bits[:-1]))
        new_state_string = ""

        for bit in new_state_bits:
            new_state_string += str(bit)

        new_state = int(new_state_string, 2)

        return new_state
    
# the main encoding function that takes the message bits and produces the encoded bits
    def encode(self, bits, terminate=True):
        bits = np.array(bits, dtype=np.uint8)

        if terminate:
            # terminate the encoded sequence by feeding K-1 trailing zeros
            zero_tail = np.zeros(self.memory, dtype=np.uint8)
            bits = np.concatenate((bits, zero_tail))
        state = 0

        encoded_bits = []

        for bit in bits: # for each input bit calculate the output bits and the next state
            output = self.output_bits(bit, state)
            for output_bit in output:
                encoded_bits.append(output_bit)
            state = self.next_state(bit, state)

        return np.array(encoded_bits, dtype=np.uint8)

    def state_to_bits(self, state): #converting the state number to its binary representation as an array of bits
        state_bits_string = format(state, f"0{self.memory}b")
        return np.array([int(bit) for bit in state_bits_string], dtype=np.uint8)
    


    def viterbi_decode(self, bits, terminate=True):
        # placeholder for viterbi decoding implementation
        pass