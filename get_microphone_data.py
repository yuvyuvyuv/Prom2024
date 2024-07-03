import sys
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt, spectrogram, find_peaks
import matplotlib.pyplot as plt
import matplotlib as mlt
from ipywidgets import widgets, interact, fixed
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import time
from scipy.signal import savgol_filter
from tqdm import tqdm, trange
from bitstring import BitArray


# Parameters
# TODO - change duration to be dynamic
duration = 7  # Duration of recording in seconds (slightly longer than the actual broadcast)

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
    b, a = butter(N=4, Wn=low, btype='highpass')
    return filtfilt(b, a, data)

# Function to calculate and plot spectrogram with bit detection lines
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import spectrogram

def calculate_spectrogram(data, fs, start_idx=0, end_idx=0, bit_positions=None, nperseg=80, noverlap=30):
    # Calculate the spectrogram
    f, t, Sxx = spectrogram(data, fs, nperseg=nperseg, noverlap=noverlap)
    Sxx += 1e-10  # Add a small constant to avoid log of zero

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    psm = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    fig.colorbar(psm, ax=ax, label='Intensity [dB]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    ax.set_title(f'Spectrogram ({cutoff_low}-{cutoff_high} Hz)')
    ax.set_ylim(cutoff_low, cutoff_high)

    start_line = ax.axvline(x=start_idx / fs, color='green', linestyle='--', label='Start')
    end_line = ax.axvline(x=end_idx / fs, color='blue', linestyle='--', label='End')
    if bit_positions is not None:
        for bit_time in bit_positions:
            ax.axvline(x=bit_time / fs, color='red', linestyle='--')
    ax.legend()

    ax_slider_start = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    start_slider = Slider(ax_slider_start, 'Start', 0, len(data) / fs, valinit=start_idx / fs)

    def update_start(val):
        start_idx = int(start_slider.val * fs)
        start_line.set_xdata([start_idx / fs])
        fig.canvas.draw_idle()

    start_slider.on_changed(update_start)

    ax_slider_end = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    end_slider = Slider(ax_slider_end, 'End', 0, len(data) / fs, valinit=end_idx / fs)

    def update_end(val):
        end_idx = int(end_slider.val * fs)
        end_line.set_xdata([end_idx / fs])
        fig.canvas.draw_idle()

    end_slider.on_changed(update_end)
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Recalculate', color='lightgoldenrodyellow', hovercolor='0.975')

    def recalculate(event):
        new_start_idx = int(start_slider.val * fs)
        new_end_idx = int(end_slider.val * fs)
        print(f"Recalculating with new start index: {new_start_idx}")
        # Here, you can call your data processing and bit decoding functions with the new start index
        # For example:
        # bits = decode_bits(data[new_start_idx:end_idx], fs, bit_duration, f1, f0)
        # print(f"Decoded bits: {''.join(bits)}")
        
        bit_positions = np.arange(new_start_idx, new_end_idx, int(bit_duration*fs))
        amplitude_ratio = calculate_amplitude_ratio(data, new_start_idx,fs)
        print(f"Amplitude Ratio: {amplitude_ratio}")

        bits = decode_bits(data,bit_positions,fs,amplitude_ratio)
        print(f"{''.join(bits[::2])}\n{''.join(bits[1::2])}")
        print(f"{''.join(bits)}")
        write_to_file(''.join(bits))
        calculate_spectrogram(data, fs, new_start_idx, new_end_idx, bit_positions=bit_positions)

    button.on_clicked(recalculate)

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
    # t[-len(t)//2:]
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
    corr1 = np.correlate(segment, ref_wav1, mode='valid') * 0.7

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

    for i, start in enumerate(tqdm(bit_positions)):
        # amplitude_ratio *= (len(bit_positions) - i) / len(bit_positions)
        # amplitude_ratio = max(amplitude_ratio, 1)
        segment = data[start : start + num_samples_per_bit]
        if len(segment) == 0:
            continue
        bits.append(detect_frequency_jeeves(segment,fs,amplitude_ratio))
    return bits

def hamming_decode(data):
    decoded_data = ''
    for i in range(0, len(data), 7):
        byte = data[i:i+7]
        p1 = int(byte[0])
        p2 = int(byte[1])
        d1 = int(byte[2])
        p3 = int(byte[3])
        d2 = int(byte[4])
        d3 = int(byte[5])
        d4 = int(byte[6])

        c1 = (p1 + d1 + d2 + d4) % 2
        c2 = (p2 + d1 + d3 + d4) % 2
        c3 = (p3 + d2 + d3 + d4) % 2

        error_pos = c1 + (c2 << 1) + (c3 << 2)

        if error_pos != 0:
            byte = byte[:error_pos-1] + str((int(byte[error_pos-1]) + 1) % 2) + byte[error_pos:]

        decoded_data += byte[2] + byte[4] + byte[5] + byte[6]
    return decoded_data

def write_to_file(bit_str: str) -> None:
    with open("output.txt", "wb") as output:
        b = BitArray(bin=bit_str)
        b.tofile(output)

def display_stats(data, filtered_data, corr1, start_idx, fs):
    plt.figure(figsize=(10, 8))  # Adjust the figure size if needed
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
    plt.plot(np.arange(len(corr1)) / fs, np.abs(corr1))
    plt.axvline(x=start_idx / fs, color='r', linestyle='--', label='Average Correlation')
    plt.title('Correlation with Preamble')
    plt.xlabel('Time (s)')
    plt.ylabel('Correlation')
    plt.legend()
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
    bits = bits[len(PREAMBLE):-len(POSTAMBLE)]
    bits = hamming_decode(bits)
    print(f"{''.join(bits[::2])}\n{''.join(bits[1::2])}")
    print(f"{''.join(bits)}")
    write_to_file(''.join(bits))
    calculate_spectrogram(filtered_data, fs, start_idx, end_idx, bit_positions=bit_positions)


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
