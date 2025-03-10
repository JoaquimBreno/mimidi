import pytorch_lightning as pl
import torch
from models.seq2seq import Seq2SeqModule
from data.midi_datamodule import MidiDataModule
import argparse
import os

def train_model(config):
    # Initialize the data module
    midi_data_module = MidiDataModule(
        midi_files=config['midi_files'],
        context_size=config['context_size'],
        description_flavor=config['description_flavor'],
        batch_size=config['batch_size']
    )

    # Initialize the model
    model = Seq2SeqModule(
        d_model=config['d_model'],
        d_latent=config['d_latent'],
        n_codes=config['n_codes'],
        n_groups=config['n_groups'],
        context_size=config['context_size'],
        lr=config['lr'],
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers']
    )

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        gpus=config['gpus'],
        progress_bar_refresh_rate=20
    )

    # Train the model
    trainer.fit(model, midi_data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Seq2Seq model.")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to the config file.')
    args = parser.parse_args()

    # Load configuration
    config = {}  # Load your configuration from the YAML file here

    train_model(config)