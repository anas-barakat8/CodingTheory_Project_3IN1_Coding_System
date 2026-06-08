# This file will define the project experiments required by the prof
import parameters as params
import hamming
import channel
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
    print("BSC TEST")
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