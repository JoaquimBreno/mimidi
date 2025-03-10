# MIMIDI Seq2Seq Model

This project implements a simplified Seq2Seq model designed for processing MIDI data. The model utilizes MIDI token embeddings and input tokens, providing a streamlined architecture while maintaining essential functionality for encoding and decoding sequences.

## Project Structure

The project is organized as follows:

```
midi-seq2seq
├── src
│   ├── models
│   │   ├── __init__.py
│   │   ├── seq2seq.py          # Simplified Seq2Seq model implementation
│   │   └── group_embedding.py   # Handles MIDI token embeddings
│   ├── data
│   │   ├── __init__.py
│   │   ├── midi_dataset.py      # Loads and processes MIDI data
│   │   └── midi_datamodule.py   # Manages data loading and batching
│   ├── utils
│   │   ├── __init__.py
│   │   ├── vocab.py             # Vocabulary management for MIDI tokens
│   │   └── constants.py         # Defines special token identifiers
│   └── train.py                 # Entry point for training the model
├── configs
│   └── default_config.yaml      # Default configuration settings
├── tests
│   ├── __init__.py
│   ├── test_model.py            # Unit tests for the Seq2Seq model
│   └── test_dataset.py          # Unit tests for the MIDI dataset
├── requirements.txt             # Project dependencies
├── setup.py                     # Setup script for the project
└── README.md                    # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd midi-seq2seq
pip install -r requirements.txt
```

## Usage

To train the Seq2Seq model, run the following command:

```bash
python src/train.py
```

Make sure to adjust the configuration settings in `configs/default_config.yaml` as needed.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.# mimidi
