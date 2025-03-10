from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .midi_dataset import MidiDataset

class MidiDataModule(LightningDataModule):
    def __init__(self, midi_files, batch_size=32, context_size=512, num_workers=4):
        super().__init__()
        self.midi_files = midi_files
        self.batch_size = batch_size
        self.context_size = context_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        self.train_dataset = MidiDataset(self.midi_files, split='train', context_size=self.context_size)
        self.val_dataset = MidiDataset(self.midi_files, split='val', context_size=self.context_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)