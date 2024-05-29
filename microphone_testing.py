import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fftpack import fft

# Parameters
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1              # Number of audio channels
RATE = 44100              # Sample rate (samples per second)
CHUNK = 1024              # Number of frames per buffer

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Prepare the plot
fig, ax = plt.subplots()
x = np.linspace(0, RATE / 2, CHUNK // 2)  # Frequency range for the x-axis
line, = ax.plot(x, np.random.rand(CHUNK // 2))
ax.set_ylim(0, 1)
ax.set_xlim(0, RATE / 2)

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
    
    # Update the line data
    line.set_ydata(yf)
    return line,

# Animate the plot
ani = FuncAnimation(fig, update, interval=50)

# Display the plot
plt.show()

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()
