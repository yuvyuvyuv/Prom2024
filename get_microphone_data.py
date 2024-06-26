import sys
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt, spectrogram, find_peaks
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import Slider

from scipy.io import wavfile


# Parameters
# TODO - change duration to be dynamic
duration = 10  # Duration of recording in seconds (slightly longer than the actual broadcast)
fs = 44000  # Sampling frequency

bps = 50 # bits per secondes
bit_duration = 1/bps  # Duration of each bit in seconds
cutoff_low = 18000
cutoff_high = 22000
high = 21000
low = 19000

# TODO!! - change to relevant paramter
amplitude_threshold = 0.1  # Constant amplitude threshold for detecting bits


def bandpass_filter(data, cutoff_low, cutoff_high, fs):
    # Function to design a bandpass filter
    nyquist = 0.5 * fs
    low = cutoff_low / nyquist
    high = cutoff_high / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='bandpass')
    return filtfilt(b, a, data)

# Function to calculate and plot spectrogram with bit detection lines
def calculate_spectrogram(data, fs, start_idx=0, end_idx=0, bit_positions=0,show=True):
    
    # TODO!! - calculate relevant nprseg and noverlap
    f, t, Sxx = spectrogram(data, fs, nperseg=128, noverlap=64)
    
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
    
    # Add lines for bit positions
    # for bit_time in bit_positions:
    #     plt.axvline(x=bit_time, color='red', linestyle='--')
    print(start_idx / fs,end_idx / fs)
    plt.axvline(x=start_idx / fs, color='green', linestyle='--', label='Start')
    plt.axvline(x=end_idx / fs, color='blue', linestyle='--', label='End')
    plt.legend()
    if show:
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




# Function to detect the start and end of the broadcast using amplitude threshold
def detect_start_and_end(data, fs, bit_duration, amplitude_threshold):
    num_samples_per_bit = int(bit_duration * fs)
    start_idx = 0
    end_idx = len(data)

    for i in range(0, len(data) - num_samples_per_bit, num_samples_per_bit):
        segment = data[i:i + num_samples_per_bit]
        fft_segment = np.abs(np.fft.rfft(segment))
        intensity_20000 = fft_segment[int(20000 * len(fft_segment) / (fs / 2))]
        intensity_19000 = fft_segment[int(19000 * len(fft_segment) / (fs / 2))]
        
        if intensity_20000 > amplitude_threshold and intensity_20000 > intensity_19000:
            next_segment = data[i + num_samples_per_bit:i + 2 * num_samples_per_bit]
            next_fft_segment = np.abs(np.fft.rfft(next_segment))
            intensity_19000 = next_fft_segment[int(19000 * len(fft_segment) / (fs / 2))]
            intensity_19000 = next_fft_segment[int(19000 * len(fft_segment) / (fs / 2))]
            
            if intensity_19000 > amplitude_threshold and intensity_20000 < intensity_19000:
                start_idx = i + num_samples_per_bit // 3
                break
    
    for i in range(len(data) - num_samples_per_bit, 0, -num_samples_per_bit):
        segment = data[i - num_samples_per_bit:i]
        fft_segment = np.abs(np.fft.rfft(segment))
        intensity_20000 = fft_segment[int(20000 * len(fft_segment) / (fs / 2))]
        
        if intensity_20000 > amplitude_threshold:
            end_idx = i
            break

    return start_idx, end_idx


def detect_signal_start(threshold,signal, preamble_bits, fs,bit_duration, high,low):
    
    preamble = encode_data(preamble_bits,bit_duration,fs,high,low)
    preamble_length = len(preamble)
    signal_length = len(signal)
    
    # Ensure the preamble matches the length of the signal segment
    if signal_length > preamble_length:
        preamble_extended = np.pad(preamble, (0, signal_length - preamble_length), mode='constant')
    else:
        preamble_extended = preamble[:signal_length]


    # Cross-correlation using FFT
    corr = np.fft.fft(signal) * np.conj(np.fft.fft(np.flip(preamble_extended)))
    corr = np.fft.ifft(corr)
    """ 
    # Find correlation index
    non_zero_corr = corr[corr > 0.1]
    avg_corr = np.mean(non_zero_corr)

    fig, ax = plt.subplots(figsize=(15, 5))
    plt.subplots_adjust(left=0.1, bottom=0.25)
    line, = ax.plot(corr, label='Cross-correlation')
    avg_line = ax.axhline(y=avg_corr, color='r', linestyle='--', label='Average Correlation')
    peaks, _ = find_peaks(corr, height=avg_corr)
    peak_line, = ax.plot(peaks, corr[peaks], 'x', label='Detected Peaks')
    ax.legend()

    axcolor = 'lightgoldenrodyellow'
    ax_thresh = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
    slider_thresh = Slider(ax_thresh, 'Threshold', 0.1 * avg_corr, 2 * avg_corr, valinit=avg_corr)

    def update(val):
        threshold = slider_thresh.val
        avg_line.set_ydata(threshold)
        peaks, _ = find_peaks(corr, height=threshold)
        peak_line.set_data(peaks, corr[peaks])
        fig.canvas.draw_idle()

    slider_thresh.on_changed(update)
    
    plt.show()
"""
    # Use the threshold value from the slider for peak detection

    start_index = np.argmax(corr > threshold)  # Take the first peak
    
    return start_index - preamble_length, corr

# Function to create a sine wave for a given frequency
def create_tone(freq, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return tone

# Function to encode data as an audio signal
def encode_data(data, bit_duration, fs, f1, f0):
    signal = np.array([])
    for bit in data:
        if bit == 1:
            bit_tone = create_tone(f1, bit_duration, fs)
        else:
            bit_tone = create_tone(f0, bit_duration, fs)
        signal = np.concatenate((signal, bit_tone))
    return signal



# Function to decode bits from the audio data
def decode_bits(data, start_idx, fs, bit_duration):
    num_samples_per_bit = int(bit_duration * fs)
    bits = []
    bit_positions = []

    for i in range(start_idx, len(data) - num_samples_per_bit, num_samples_per_bit):
        segment = data[i:i + num_samples_per_bit]
        if len(segment) == 0:
            continue
        
        fft_segment = np.abs(np.fft.rfft(segment))
        intensity_19000 = fft_segment[int(19000 * len(fft_segment) / (fs / 2))]
        intensity_20000 = fft_segment[int(20000 * len(fft_segment) / (fs / 2))]
        
        bit_positions.append(i / fs + bit_duration)
        
        if intensity_20000 > intensity_19000:
            bits.append('1')
        else:
            bits.append('0')

    return bits, bit_positions

def display_stats(data,filtered_data,corr1,true1,fs):
    # Plot original CSS signal
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(data)) / fs, data)
    plt.title('Original CSS Signal')
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
    plt.axvline(x=true1/fs, color='r', linestyle='--', label='Average Correlation')
    plt.title('Correlation with Preamble')
    plt.xlabel('Time (s)')
    plt.ylabel('Correlation')
    plt.show()

# Main function
def main(source, filename=None, fs=fs):
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
    
    # Calculate the number of samples received per second
    total_samples = len(data)
    samples_per_second = total_samples / elapsed_time
    print(f"Total samples recorded: {total_samples}")
    print(f"Samples per second: {samples_per_second:.2f}")
    
    # Apply bandpass filter
    filtered_data = bandpass_filter(data, cutoff_low, cutoff_high, fs)
    
    # Detect start and end of the broadcast
    start_idx, end_idx = detect_start_and_end(filtered_data, fs, bit_duration, amplitude_threshold)
    print(f"Detected broadcast start at sample index: {start_idx}, at time: {start_idx / samples_per_second}")
    print(f"Detected broadcast end at sample index: {end_idx}, at time: {end_idx / samples_per_second}")
    
    preamble_bits = [1,0,1,0]
    true1, corr1 = detect_signal_start(.8,filtered_data,[1,0,1],fs,bit_duration,high,low)
    print(true1,true1/fs)
    display_stats(data,filtered_data,corr1,true1,fs)
    # Trim the data 2 bit durations before the start
    start_idx = max(start_idx, 0)
    trimmed_data = filtered_data[start_idx:end_idx]

    calculate_spectrogram(filtered_data, fs, true1, true1 + bit_duration*fs/2)

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







def fft_cross_correlation(x, y):
    N = len(x) + len(y) - 1
    X = np.fft.fft(x, N)
    Y = np.fft.fft(y, N)
    corr = np.fft.ifft(X * np.conj(Y))
    return np.abs(corr)

def decode_bfsk_signal(filtered_signal, fs, f0, f1, preamble_signal, bit_duration, threshold_factor=1.5):

    # Step 4: Detect the start of the preamble using cross-correlation with FFT
    correlation = fft_cross_correlation(filtered_signal, preamble_signal)
    peaks, _ = find_peaks(correlation, height=np.max(correlation) * 0.5)
    start_index = peaks[0]  # Take the first peak
												 
    start_time = start_index / fs
    print("Start of preamble detected at:", start_time, "seconds")

    # Step 5: Decode the signal using hypothesis testing
    samples_per_bit = int(bit_duration * fs)
    num_bits = (len(filtered_signal) - start_index) // samples_per_bit

    decoded_bits = []
    for i in range(num_bits):
        start = start_index + i * samples_per_bit
        end = start + samples_per_bit
        segment = filtered_signal[start:end]
        
        corr_f0 = np.sum(np.sin(2 * np.pi * f0 * np.arange(0, bit_duration, 1/fs)) * segment)
        corr_f1 = np.sum(np.sin(2 * np.pi * f1 * np.arange(0, bit_duration, 1/fs)) * segment)
        
        if corr_f1 > corr_f0 * threshold_factor:
            decoded_bits.append(1)
        else:
            decoded_bits.append(0)

    return decoded_bits


