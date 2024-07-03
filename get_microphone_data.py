import sys
# import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt, spectrogram, find_peaks
import matplotlib.pyplot as plt

import time
from scipy.signal import savgol_filter
from tqdm import tqdm, trange
from bitstring import BitArray


# Parameters
# TODO - change duration to be dynamic
duration = 2  # Duration of recording in seconds (slightly longer than the actual broadcast)

bps = 50 # bits per secondes
bit_duration = 1/bps  # Duration of each bit in seconds
cutoff_low = 18000
cutoff_high = 22000
f1 = 21000
f0 = 19000


def fft_cross_correlation(x, y):
    N = len(x) + len(y) - 1
    X = np.fft.fft(x, N)
    Y = np.fft.fft(y, N)
    corr = np.fft.ifft(X * np.conj(Y))
    return np.abs(corr)

def highpass_filter(data, cutoff_low, cutoff_high,fs):
    # Function to design a bandpass filter
    nyquist = 0.5 * fs
    low = cutoff_low / nyquist
    high = cutoff_high / nyquist
    b, a = butter(N=4, Wn=low, btype='highpass')
    return filtfilt(b, a, data)

# Function to calculate and plot spectrogram with bit detection lines
def calculate_spectrogram(data,bits,fs, bit_positions=0,nperseg=128, noverlap=64):
    start_idx = bit_positions[0] 
    end_idx = bit_positions[-1]
    # TODO!! - calculate relevant nprseg and noverlap
    f, t, Sxx = spectrogram(data, fs, nperseg=nperseg, noverlap=noverlap)
    
    # Add a small constant to avoid log of zero
    Sxx += 1e-10
    plt.style.use('dark_background')
    # Plot the spectrogram
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(f'Spectrogram ({cutoff_low}-{cutoff_high} Hz)')
    plt.colorbar(label='Intensity [dB]')
    plt.ylim(cutoff_low, cutoff_high)
    plt.xlim((start_idx/fs-2*bit_duration)/1.4, end_idx/fs+3*bit_duration)
    # Add lines for bit positions
    for i,bit_time in enumerate(bit_positions):
        plt.axvline(x=bit_time/fs, color='red', linestyle='--')
        plt.text(bit_time/fs, (f1+f0)/2,bits[i])
    plt.axvline(x=start_idx / fs, color='green', linestyle='--', label='Start')
    plt.axvline(x=end_idx / fs, color='blue', linestyle='--', label='End')
    plt.gca().margins(y=0.2)
    plt.legend()
    plt.show()

# Function to get microphone data
def record_audio(duration, fs):
    print("Recording...")
    start_time = time.time()
    data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Recording complete. Duration: {elapsed_time:.2f} seconds")
    return data.flatten(), elapsed_time

def detect_signal_edge(signal, preamble_bits, postamble_bits,fs):
    preamble = encode_data(preamble_bits,fs)
    postamble = encode_data(postamble_bits,fs)
    # Cross-correlation using FFT
    start_corr = fft_cross_correlation(signal,preamble)
    end_corr = fft_cross_correlation(signal,postamble)

    window_size = int(bit_duration*fs/4)

    start_smoothed_corr = savgol_filter(start_corr,window_size,3)
    end_smoothed_corr = np.flip(savgol_filter(end_corr,window_size,3))
    start_peaks, _ = find_peaks(start_smoothed_corr,width = bit_duration*fs/4,height=1)
    end_peaks, _ = find_peaks(end_smoothed_corr,width = bit_duration*fs/4,height=1)
    start_idx = 0
    if len(start_peaks) != 0:
        start_idx = start_peaks[0]
    end_idx = len(signal)
    if len(end_peaks) != 0:
        end_idx = len(signal) - end_peaks[0]
    return start_idx, end_idx, start_corr,end_corr

# Function to create a sine wave for a given frequency
def create_tone(freq, duration,fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return tone

# Function to encode data as an audio signal
def encode_data(data,fs):
    signal = np.array([])
    for bit in data:
        if bit == 1:
            bit_tone = create_tone(f1, bit_duration,fs)
        else:
            bit_tone = create_tone(f0, bit_duration,fs)
        signal = np.concatenate((signal, bit_tone))
    return signal

def detect_frequency_yuv(segment, fs,amplitude_ratio = 1):
    # Generate reference sine waves for correlation
    t = np.arange(len(segment)) / fs
    ref_wav0 = np.sin(2 * np.pi * f0 * t)
    ref_wav1 = np.sin(2 * np.pi * f1 * t)

    # Normalize the segment and reference signals
    segment_norm = segment / np.linalg.norm(segment)
    ref_wav0_norm = ref_wav0 / np.linalg.norm(ref_wav0)
    ref_wav1_norm = ref_wav1 / np.linalg.norm(ref_wav1)

    # val0 = np.linalg.norm(segment_norm - ref_wav0_norm)
    # val1 = np.linalg.norm(segment_norm - ref_wav1_norm)
    # plt.plot(segment_norm)
    # plt.plot(ref_wav0_norm)
    # plt.plot(ref_wav1_norm)
    window_length = 150
    cor0 = fft_cross_correlation(segment_norm,ref_wav0_norm)
    cor1 = fft_cross_correlation(segment_norm,ref_wav1_norm)
    cor0s = savgol_filter(cor0,window_length,3)
    cor1s = savgol_filter(cor1,window_length,3)
    # plt.plot(cor0s)
    # plt.plot(cor1s)
    # val0 = np.dot(ref_wav0_norm, segment_norm) #/ (np.linalg.norm(ref_wav0_norm) * np.linalg.norm(segment_norm))
    # val1 = np.dot(ref_wav1_norm, segment_norm) *amplitude_ratio#/ (np.linalg.norm(ref_wav1_norm) * np.linalg.norm(segment_norm))
    # print(val0,val1)
    val0 = np.mean(cor0s)
    val1 = np.mean(cor1s)
    
    if val0 > val1:
        return "0"
    else:
        return "1"


def calculate_amplitude_ratio(filtered_data, start_idx,fs):
    num_samples_per_bit = int(bit_duration * fs)
    segment_0 = filtered_data[start_idx: start_idx + num_samples_per_bit]
    segment_1 = filtered_data[start_idx + num_samples_per_bit: start_idx + 2 * num_samples_per_bit]

    t = np.arange(len(segment_0)) / fs
    ref_wav0 = np.sin(2 * np.pi * f0 * t)
    ref_wav1 = np.sin(2 * np.pi * f1 * t)

    corr0_0 = np.correlate(segment_0, ref_wav0, mode='valid')
    corr0_1 = np.correlate(segment_1, ref_wav1, mode='valid')

    amplitude_0 = np.max(corr0_0)
    amplitude_1 = np.max(corr0_1)

    ratio = amplitude_0 / amplitude_1
    return ratio

def detect_frequency_jeeves(segment, fs,amplitude_ratio):
    t = np.arange(len(segment)) / fs
    ref_wav0 = np.sin(2 * np.pi * f0 * t)
    ref_wav1 = np.sin(2 * np.pi * f1 * t)

    corr0 = np.correlate(segment, ref_wav0, mode='valid')
    corr1 = np.correlate(segment, ref_wav1, mode='valid')

    max_corr0 = np.max(corr0)
    max_corr1 = np.max(corr1) * amplitude_ratio

    if max_corr0 > max_corr1:
        return '0'
    else:
        return '1'

# Function to decode bits from the audio data
def decode_bits(data, bit_positions,fs, amplitude_ratio):
    num_samples_per_bit = int(bit_duration * fs)
    bits = []

    for start in tqdm(bit_positions):
        segment = data[start : start + num_samples_per_bit]
        if len(segment) == 0:
            continue
        bits.append(detect_frequency_jeeves(segment,fs,amplitude_ratio))
    return bits

def write_to_file(bit_str: str) -> None:
    with open("output.txt", "wb") as output:
        b = BitArray(bin=bit_str)
        b.tofile(output)

def display_stats(data,filtered_data,corr,start_idx,fs):
    # Plot original CSS signal
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(data)) / fs, data)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot filtered signal
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(filtered_data)) / fs, filtered_data)
    plt.title('Filtered Signal (18-22 kHz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot correlation result
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(corr)) / fs, np.abs(corr))
    plt.axvline(x=start_idx/fs, color='r', linestyle='--', label='Average Correlation')
    plt.title('Correlation with Preamble')
    plt.xlabel('Time (s)')
    plt.ylabel('Correlation')
    plt.show()


# Main function
def main(source, filename=None):
    fs = 44000
    if source == "live":
        data, elapsed_time = record_audio(duration, fs)
    elif source == "file":
        if filename is None:
            raise ValueError("Filename must be provided when source is 'file'")
        data, fs = sf.read(filename)
        elapsed_time = len(data) / fs
    elif source == "live_save":
        if filename is None:
            raise ValueError("Filename must be provided when source is 'live_save'")
        data, elapsed_time = record_audio(duration, fs)
        sf.write(filename, data, fs)
        print(f"Recording saved to {filename}")
    else:
        raise ValueError("Invalid source. Must be 'live', 'file', or 'live_save'")

    print(f"bps: {bps},   fs: {fs:.2f}")
    
    filtered_data = highpass_filter(data, cutoff_low, cutoff_high,fs)
    start_preamble_bits = [1,0,1,0,1]
    end_preamble_bits = start_preamble_bits

    start_idx, end_idx, start_corr,end_corr = detect_signal_edge(filtered_data,
                    start_preamble_bits,end_preamble_bits,fs)
    display_stats(data,filtered_data,start_corr,start_idx,fs)
    bit_positions = np.arange(start_idx, end_idx, int(bit_duration*fs))
    amplitude_ratio = calculate_amplitude_ratio(filtered_data, start_idx,fs)
    print(f"Amplitude Ratio: {amplitude_ratio}")

    bits = decode_bits(filtered_data,bit_positions,fs,amplitude_ratio)
    print(''.join(bits))
    write_to_file(''.join(bits))
    calculate_spectrogram(filtered_data, bits,fs,bit_positions,nperseg=256,noverlap=16)


if __name__ == "__main__":
    # Parse command line arguments if running from command line
    if len(sys.argv) < 2:
        print("Usage: python your_script.py source [filename]")
        sys.exit(1)

    source = sys.argv[1]
    filename = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        main(source, filename)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
