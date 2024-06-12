import numpy as np
import sounddevice as sd

# Parameters
fs = 44100  # Sampling frequency
duration = 5  # Duration of the sound in seconds
freq1 = 20000  # Frequency 1 in Hz
freq2 = 18000  # Frequency 2 in Hz
switch_rate = 100  # Number of times to switch between frequencies per second

# Create the on-off pattern
def create_switch_pattern(rate, fs, duration):
    # Calculate the length of each on/off segment in samples
    segment_length = fs // (2 * rate)
    # Create one cycle of switching pattern
    one_cycle = np.concatenate([np.ones(segment_length), np.zeros(segment_length)])
    # Repeat the pattern to fill the entire duration
    num_cycles = int(duration * rate)
    pattern = np.tile(one_cycle, num_cycles)
    # Trim the pattern to match the exact number of samples needed
    return pattern[:int(duration * fs)]

# Create the noise for a given frequency
def create_noise(freq, fs, duration):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    noise = 0.5 * np.sin(2 * np.pi * freq * t)
    return noise

# Main function
def main():
    pattern = create_switch_pattern(switch_rate, fs, duration)
    noise1 = create_noise(freq1, fs, duration)
    noise2 = create_noise(freq2, fs, duration)
    
    # Ensure both arrays have the same length
    min_length = min(len(pattern), len(noise1), len(noise2))
    pattern = pattern[:min_length]
    noise1 = noise1[:min_length]
    noise2 = noise2[:min_length]
    
    # Create the alternating signal
    signal = np.where(pattern, noise1, noise2)
    sd.play(signal, fs)
    sd.wait()

if __name__ == "__main__":
    main()
