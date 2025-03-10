from torch.utils.data import Dataset
import os
import numpy as np
import torch

class MidiDataset(Dataset):
    def __init__(self, midi_files, vocab, max_length=256):
        self.midi_files = midi_files
        self.vocab = vocab
        self.max_length = max_length
        self.data = self.load_data()

    def load_data(self):
        sequences = []
        for file in self.midi_files:
            if os.path.exists(file):
                # Load MIDI file and convert to token sequence
                # This is a placeholder for actual MIDI loading logic
                token_sequence = self.midi_to_tokens(file)
                sequences.append(token_sequence)
        return sequences

    def midi_to_tokens(self, file):
        # Placeholder for MIDI to token conversion logic
        # For now, we will return a random sequence of tokens
        return np.random.randint(0, len(self.vocab), size=self.max_length).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        # Pad sequence to max_length
        if len(sequence) < self.max_length:
            sequence += [self.vocab.to_i('<PAD>')] * (self.max_length - len(sequence))
        return torch.tensor(sequence, dtype=torch.long)