import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from scipy.fftpack import fft

# Parameters
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1              # Number of audio channels
RATE = 44100              # Sample rate (samples per second)
CHUNK = 1024              # Number of frames per buffer
TARGET_HIGH_FREQ = 20154       # Target frequency to detect spikes around
ERROR_MARGIN = 400        # Error margin for the target frequency
TARGET_LOW_FREQ = 19154       # Target frequency to detect spikes around

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Calculate frequency bins
freqs = np.fft.fftfreq(CHUNK, 1 / RATE)[:CHUNK // 2]

# Find the index range for 19,000 Hz to 25,000 Hz
min_freq_idx = np.where(freqs >= 0)[0][0]
max_freq_idx = np.where(freqs <= 25000)[0][-1]

# Prepare the plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # Adjust plot to make room for slider
x = freqs[min_freq_idx:max_freq_idx+1]  # Frequency range for the x-axis
line, = ax.plot(x, np.random.rand(len(x)))
ax.set_ylim(0, 1)
ax.set_xlim(0, 25000)

# Add a slider for threshold adjustment
axcolor = 'lightgoldenrodyellow'
ax_threshold = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
threshold_slider = Slider(ax_threshold, 'Threshold', 0.001, 0.1, valinit=0.1)

high_target_range = np.where((freqs >= (TARGET_HIGH_FREQ - ERROR_MARGIN)) & (freqs <= (TARGET_HIGH_FREQ + ERROR_MARGIN)))[0]
# Update function for the animation
def update(frame):
    # Read audio data from the stream
    data = stream.read(CHUNK, exception_on_overflow=False)
    # Convert data to numpy array
    audio_data = np.frombuffer(data, dtype=np.int16)
    # Normalize the data
    audio_data = audio_data / np.max(np.abs(audio_data))
    # Perform FFT
    yf = fft(audio_data)
    # Take the magnitude of the FFT and normalize
    yf = np.abs(yf[:CHUNK // 2]) * 2 / CHUNK
    
    # Extract the relevant part of the FFT result
    yf = yf[min_freq_idx:max_freq_idx+1]
    
    # Update the line data
    line.set_ydata(yf)
    
    # Find spikes in the target frequency range
    high_target_spike = np.max(yf[high_target_range])
    # Find spikes in the target frequency range
    low_target_range = np.where((freqs >= (TARGET_HIGH_FREQ - ERROR_MARGIN)) & (freqs <= (TARGET_HIGH_FREQ + ERROR_MARGIN)))[0]
    low_target_spike = np.max(yf[low_target_range])
    
    # Print spike value if found
    if high_target_spike > threshold_slider.val:  # Use slider value as threshold
        print(1)
    elif high_target_spike > threshold_slider.val:  # Use slider value as threshold
        print(0)
    else:
        print("NONE")
    
    return line

# Animate the plot
ani = FuncAnimation(fig, update, interval=50)

# Display the plot
plt.show()

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()
