# This file will define the project experiments required by the prof
import parameters as params
import hamming
import channel
import convolutional
import concatenated
from utilities import *


def hamming_only():
    print("=" * 50)
    print("HAMMING CODE TEST")
    print("=" * 50)

    hamming_encoder = hamming.HammingCode(params.H, params.k)

    message_bits = text_to_bits(params.personal_message)
    encoded_bits = hamming_encoder.encode(message_bits)
    decoded_bits = hamming_encoder.decode(encoded_bits)

    expected_length = int(len(message_bits) / hamming_encoder.k * hamming_encoder.n)

    print("Original message:")
    print(" ", params.personal_message)
    print()

    print("Hamming parameters:")
    print("  k =", hamming_encoder.k, "message bits per block")
    print("  n =", hamming_encoder.n, "encoded bits per block")
    print()

    print_bits_summary("Original message bits:", message_bits)
    print_bits_summary("Encoded bits:", encoded_bits)

    print("Encoding test:")
    print("  Expected encoded length:", expected_length)
    print("  Actual encoded length:  ", len(encoded_bits))
    print("  Passed:", len(encoded_bits) == expected_length)
    print()

    print_bits_summary("Decoded bits:", decoded_bits)

    print("Decoding test:")
    print("  Passed:", np.array_equal(message_bits, decoded_bits))
    print()
    """
    print("-" * 50)
    print("SINGLE-BIT ERROR CORRECTION TEST")
    print("-" * 50)

    corrupted_bits = encoded_bits.copy()

    error_position = 2
    corrupted_bits[error_position] = np.mod(corrupted_bits[error_position] + 1, 2)

    decoded_corrupted_bits = hamming_encoder.decode(corrupted_bits)

    print("Corruption:")
    print("  Flipped bit position:", error_position)
    print("  Original bit:", encoded_bits[error_position])
    print("  Corrupted bit:", corrupted_bits[error_position])
    print()

    print_bits_summary("Corrupted encoded bits:", corrupted_bits)
    print_bits_summary("Decoded corrupted bits:", decoded_corrupted_bits)

    print("Single-bit error correction test:")
    print("  Passed:", np.array_equal(message_bits, decoded_corrupted_bits))
    print()

    print("=" * 50)

    

    """
    
    print("=" * 50)
    print(" HAMMING BSC TEST")
    print("=" * 50)
    for ber in params.ber_values:
        for seed in params.seeds:

            #ber = params.ber_values[3]
            #seed = params.seeds[0]
            encoded_bits = hamming_encoder.encode(message_bits)
            received_bits, errors_before = channel.Binary_Symmetric_Channel(encoded_bits, ber, seed)

            decoded_received_bits = hamming_encoder.decode(received_bits)

            errors_after = np.sum(message_bits != decoded_received_bits)
            post_decoding_ber = round(errors_after/len(message_bits),4)

            print("BSC settings:")
            print("  Channel BER:", ber)
            print("  Noise Seed:", seed)
            print()

            print("Channel errors:")
            print("  Errors before decoding:", errors_before)
            print("  Errors after decoding: ", errors_after)
            print("  Perfect decode:", errors_after == 0)
            print("  Post-decoding BER:", post_decoding_ber)
            print()

            print_bits_summary("Received bits after BSC:", received_bits)
            print_bits_summary("Decoded BSC bits:", decoded_received_bits)



def convolutional_only():
    print("=" * 50)
    print("CONVOLUTIONAL CODE TEST")
    print("=" * 50)

    convolutional_encoder = convolutional.ConvolutionalCode(params.generators, params.K, params.memory)

    message_bits = text_to_bits(params.personal_message)

    encoded_bits_terminated = convolutional_encoder.encode(message_bits, terminate=True)
    #

    number_of_generators = len(convolutional_encoder.generators)

    

    expected_length_terminated = (len(message_bits) + convolutional_encoder.memory) * number_of_generators

    print("Original message:")
    print(" ", params.personal_message)
    print()

    print("Convolutional parameters:")
    print("  Constraint length:", convolutional_encoder.constraint_length)
    print("  Memory:", convolutional_encoder.memory)
    print("  Number of states:", convolutional_encoder.num_states)
    print("  Number of generators:", number_of_generators)
    print("  Generators:")
    print(convolutional_encoder.generators)
    print()

    print_bits_summary("Original message bits:", message_bits)

    print_bits_summary("Encoded bits with termination:", encoded_bits_terminated)

    print("Encoding length test with termination:")
    print("  Expected encoded length:", expected_length_terminated)
    print("  Actual encoded length:  ", len(encoded_bits_terminated))
    print("  Passed:", len(encoded_bits_terminated) == expected_length_terminated)
    print()

    
    print("-" * 50)
    print("STATE TRANSITION TEST")
    print("-" * 50)

    state = 0

    print("Start state:", state)
    print("Start state bits:", convolutional_encoder.state_to_bits(state))
    print()

    for bit in message_bits[:8]:
        output = convolutional_encoder.output_bits(bit, state)
        next_state = convolutional_encoder.next_state(bit, state)

        print("Input bit:", bit)
        print("  Current state:", state)
        print("  Current state bits:", convolutional_encoder.state_to_bits(state))
        print("  Output bits:", output)
        print("  Next state:", next_state)
        print("  Next state bits:", convolutional_encoder.state_to_bits(next_state))
        print()

        state = next_state

    print("=" * 50)            



    decoded_bits = convolutional_encoder.viterbi_decode(encoded_bits_terminated, terminate=True)
    """
    print("-" * 50)
    print("Viterbi clean decoding test:")
    print("-" * 50)
    print("  Original length:", len(message_bits))
    print("  Encoded length with termination:", len(encoded_bits_terminated))
    print("  Decoded length: ", len(decoded_bits))
    print("  Original bits:  ", message_bits)
    print("  Encoded bits:   ", encoded_bits_terminated)
    print("  Decoded bits:   ", decoded_bits)
    print("  Passed:", np.array_equal(message_bits, decoded_bits))
    print()

    print("-" * 50)
    print("VITERBI SINGLE-BIT ERROR TEST")
    print("-" * 50)

    received_bits_with_error = encoded_bits_terminated.copy()

    error_position = 5
    received_bits_with_error[error_position] ^= 1

    decoded_error_bits = convolutional_encoder.viterbi_decode(received_bits_with_error, terminate=True)
    

    print("Original message length: ", len(message_bits))
    print("Terminated encoded length:  ", len(encoded_bits_terminated))
    print("Received with error length: ", len(received_bits_with_error))
    print("Decoded message length:    ", len(decoded_error_bits))
    print()

    print("Flipped encoded bit position:   ", error_position)
    print("Original encoded bit:     ", encoded_bits_terminated)
    print("Received bits with error:", received_bits_with_error)
    print()

    print("Original message bits:", message_bits)
    print("Decoded from error:    ", decoded_error_bits)
    print("Passed:", np.array_equal(message_bits, decoded_error_bits))
    print()
    """
    print("-" * 50)
    print(" CONVOLUTIONAL BSC TEST ")
    print("-" * 50)

    for ber in params.ber_values:
        for seed in params.seeds:


            encoded_bits = convolutional_encoder.encode(message_bits, terminate=True)
            received_bits, errors_before = channel.Binary_Symmetric_Channel(encoded_bits, ber, seed)

            decoded_received_bits = convolutional_encoder.viterbi_decode(received_bits, terminate=True)

            errors_after = np.sum(message_bits != decoded_received_bits)
            post_decoding_ber = round(errors_after/len(message_bits),4)

            print("BSC settings:")
            print("  Channel BER:", ber)
            print("  Noise Seed:", seed)
            print()

            print("Channel errors:")
            print("  Errors before decoding:", errors_before)
            print("  Errors after decoding: ", errors_after)
            print("  Perfect decode:", errors_after == 0)
            print("  Post-decoding BER:", post_decoding_ber)
            print()

            print_bits_summary("Received bits after BSC:", received_bits)
            print_bits_summary("Decoded BSC bits:", decoded_received_bits)




def conatenated():

    print("=" * 50)
    print("CONCATENATED CODE TEST")
    print("=" * 50)

    message_bits = text_to_bits(params.personal_message)

    concatenated_encoded_bits = concatenated.concatenated_encode(message_bits)    
    concateneted_expected_length = int(((len(message_bits) / params.k * params.n) + params.memory) * (len(params.generators)))

    print("  Original length:", len(message_bits))
    print("  Encoded bits length:", len(concatenated_encoded_bits))
    print("  Expected encoded length:", concateneted_expected_length)
    print("  Passed:", len(concatenated_encoded_bits) == concateneted_expected_length)
    print()
