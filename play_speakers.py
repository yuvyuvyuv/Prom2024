from bitstring import BitArray
import numpy as np
import sounddevice as sd

# Parameters
fs = 96000  # Sampling frequency
bit_duration = 0.1  # Duration of each bit in seconds
freq1 = 20000  # Frequency for '1' in Hz
freq0 = 19000  # Frequency for '0' in Hz
PREAMBLE = '10101'
POSTAMBLE = '10101'

# Function to create a sine wave for a given frequency
def create_tone(freq, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return tone

# Function to encode data as an audio signal
def encode_data(data, bit_duration, fs, freq1, freq0):
    signal = np.array([])
    for bit in data:
        if bit == '1':
            tone = create_tone(freq1, bit_duration, fs)
        elif bit == '0':
            tone = create_tone(freq0, bit_duration, fs)
        signal = np.concatenate((signal, tone))
    return signal

# Main function
def main():
    # Example data to encode
    input_fname = "input.txt"
    data = BitArray(filename=input_fname).bin  # Example bit sequence
    print(data)
    data = PREAMBLE + str(data) + POSTAMBLE
    
    # Encode the data
    signal = encode_data(data, bit_duration, fs, freq1, freq0)
    
    # Play the encoded signal
    sd.play(signal, fs)
    sd.wait()
    
    print(f"Encoded and played data: {data}")

if __name__ == "__main__":
    main()
