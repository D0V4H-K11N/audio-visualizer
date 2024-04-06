import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pytube import YouTube

def download_audio_from_youtube(youtube_url, output_file):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    stream.download(output_path='.', filename=output_file)

def visualize_audio_from_file(audio_path):
    audio_data, sampling_rate = librosa.load(audio_path)
    spectrogram = np.abs(librosa.stft(audio_data))
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sampling_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

# Example usage
youtube_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'  # Example YouTube URL
output_file = 'audio_from_youtube.wav'  # Output audio file name

download_audio_from_youtube(youtube_url, output_file)
visualize_audio_from_file(output_file)
