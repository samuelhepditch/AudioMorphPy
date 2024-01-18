import os
import torchaudio
import torch
from torch.utils.data import Dataset

def load_audio_file(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    # Here you can add more preprocessing steps if needed
    return waveform

class VoiceDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.file_names = os.listdir(directory)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.file_names[idx])
        waveform = load_audio_file(file_path)
        # Here, you should implement the logic to get the label of the voice
        # For simplicity, let's assume it's the first character of the file name
        label = int(self.file_names[idx][0])
        return waveform, label
