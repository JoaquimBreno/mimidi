import pytest
import torch
from src.models.seq2seq import Seq2SeqModule

def test_seq2seq_forward():
    model = Seq2SeqModule(d_model=128, d_latent=128, n_codes=256, n_groups=4)
    input_ids = torch.randint(0, 256, (2, 10))  # Batch size of 2, sequence length of 10
    latents = torch.randn(2, 10, 128)  # Batch size of 2, sequence length of 10, latent dimension of 128
    labels = torch.randint(0, 256, (2, 10))  # Batch size of 2, sequence length of 10

    output = model(input_ids, z=latents, labels=labels)
    
    assert output.shape == (2, 10, 256), "Output shape mismatch"

def test_seq2seq_loss():
    model = Seq2SeqModule(d_model=128, d_latent=128, n_codes=256, n_groups=4)
    input_ids = torch.randint(0, 256, (2, 10))
    latents = torch.randn(2, 10, 128)
    labels = torch.randint(0, 256, (2, 10))

    loss = model.get_loss({'input_ids': input_ids, 'latents': latents, 'labels': labels})
    
    assert loss.item() >= 0, "Loss should be non-negative"

def test_seq2seq_sample():
    model = Seq2SeqModule(d_model=128, d_latent=128, n_codes=256, n_groups=4)
    input_ids = torch.randint(0, 256, (2, 10))
    latents = torch.randn(2, 10, 128)

    sample_output = model.sample({'input_ids': input_ids, 'latents': latents}, max_length=20)
    
    assert sample_output['sequences'].shape == (2, 20), "Sampled sequences shape mismatch"
    assert sample_output['bar_ids'].shape == (2, 20), "Sampled bar_ids shape mismatch"
    assert sample_output['position_ids'].shape == (2, 20), "Sampled position_ids shape mismatch"