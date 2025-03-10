import unittest
from src.data.midi_dataset import MidiDataset

class TestMidiDataset(unittest.TestCase):

    def setUp(self):
        # Initialize the MidiDataset with a sample MIDI file path
        self.dataset = MidiDataset('path/to/sample_midi_file.mid')

    def test_load_data(self):
        # Test if the dataset loads data correctly
        data = self.dataset.load_data()
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

    def test_tokenize(self):
        # Test the tokenization of MIDI data
        tokens = self.dataset.tokenize()
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_get_item(self):
        # Test if getting an item from the dataset works correctly
        item = self.dataset[0]
        self.assertIsInstance(item, dict)
        self.assertIn('input_ids', item)
        self.assertIn('bar_ids', item)

if __name__ == '__main__':
    unittest.main()