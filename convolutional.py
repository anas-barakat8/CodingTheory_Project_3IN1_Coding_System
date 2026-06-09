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
    


    def viterbi_decode(self, received_bits, terminate=True):
        received_bits = np.array(received_bits, dtype=np.uint8)
        number_of_output_bits = len(self.generators)

        received_blocks = received_bits.reshape(-1, number_of_output_bits)

        path_metrics = np.full(self.num_states, np.inf)
        path_metrics[0] = 0

        paths = []
        for state in range(self.num_states):
            paths.append([])

        for received_block in received_blocks:
            new_path_metrics = np.full(self.num_states, np.inf)

            new_paths = []
            for state in range(self.num_states):
                new_paths.append([])

            for current_state in range(self.num_states):
                if path_metrics[current_state] == np.inf:
                    continue

                for input_bit in [0, 1]:
                    expected_output = self.output_bits(input_bit, current_state)
                    next_state = self.next_state(input_bit, current_state)

                    distance = self.hamming_distance(received_block, expected_output)
                    new_metric = path_metrics[current_state] + distance

                    if new_metric < new_path_metrics[next_state]:
                        new_path_metrics[next_state] = new_metric
                        new_paths[next_state] = paths[current_state] + [input_bit]

            path_metrics = new_path_metrics
            paths = new_paths

        if terminate:
            best_state = 0
        else:
            best_state = np.argmin(path_metrics)

        decoded_bits = np.array(paths[best_state], dtype=np.uint8)

        if terminate:
            decoded_bits = decoded_bits[: -self.memory]

        return decoded_bits

    def hamming_distance(self, bits_a, bits_b): # calculate the Hamming distance between two arrays of bits
        bits_a = np.array(bits_a, dtype=np.uint8)
        bits_b = np.array(bits_b, dtype=np.uint8)
        difference = bits_a != bits_b
        distance = np.sum(difference)
        return distance