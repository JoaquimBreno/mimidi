import os
import glob
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import miditok
import miditoolkit
import random
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Train a tokenizer on MIDI files")
    parser.add_argument(
        "--midi_dir", 
        type=str, 
        required=True,
        help="Directory containing MIDI files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="tokenizer",
        help="Output directory for tokenizer files"
    )
    parser.add_argument(
        "--tokenizer_type", 
        type=str, 
        default="REMI",
        choices=["REMI", "MIDILike", "TSD", "Structured", "CPWord", "MMM"],
        help="Tokenizer type (miditok)"
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=2048,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=1000,
        help="Number of MIDI files to sample for training (use -1 for all)"
    )
    parser.add_argument(
        "--special_tokens", 
        type=str, 
        default="[PAD],[UNK],[CLS],[SEP],[MASK]",
        help="Comma-separated special tokens"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()

def get_midi_files(directory):
    """Get all MIDI files in the directory (recursively)"""
    return glob.glob(os.path.join(directory, "**", "*.mid"), recursive=True)

def create_tokenizer(args):
    """Create a tokenizer instance based on args"""
    special_tokens = args.special_tokens.split(",")
    
    # Create tokenizer config based on type
    if args.tokenizer_type == "REMI":
        tokenizer = miditok.REMI(
            special_tokens=special_tokens,
            use_drum=True,
            use_sustained=True,
            use_chords=True,
            beat_res={(0, 4): 8, (4, 12): 4},
        )
    elif args.tokenizer_type == "MIDILike":
        tokenizer = miditok.MIDILike(
            special_tokens=special_tokens,
            use_drum=True,
            use_sustained=True
        )
    elif args.tokenizer_type == "TSD":
        tokenizer = miditok.TSD(
            special_tokens=special_tokens,
            use_drum=True,
            use_sustained=True
        )
    elif args.tokenizer_type == "Structured":
        tokenizer = miditok.Structured(
            special_tokens=special_tokens,
            use_drum=True,
            use_sustained=True,
            use_chords=True,
        )
    elif args.tokenizer_type == "CPWord":
        tokenizer = miditok.CPWord(
            special_tokens=special_tokens,
            use_drum=True,
            use_sustained=True
        )
    elif args.tokenizer_type == "MMM":
        tokenizer = miditok.MMM(
            special_tokens=special_tokens,
            use_drum=True,
            use_sustained=True
        )
    else:
        raise ValueError(f"Unknown tokenizer type: {args.tokenizer_type}")
    
    return tokenizer

def train_tokenizer(args):
    """Train a tokenizer on MIDI files"""
    print(f"Finding MIDI files in {args.midi_dir}...")
    midi_files = get_midi_files(args.midi_dir)
    print(f"Found {len(midi_files)} MIDI files")
    
    # Sample files if needed
    if args.sample_size > 0 and args.sample_size < len(midi_files):
        random.seed(args.random_seed)
        midi_files = random.sample(midi_files, args.sample_size)
        print(f"Sampled {len(midi_files)} MIDI files for training")
    
    # Create the tokenizer
    tokenizer = create_tokenizer(args)
    
    # Process MIDI files
    print("Processing MIDI files...")
    token_sequences = []
    errors = 0
    
    for midi_file in tqdm(midi_files):
        try:
            midi = miditoolkit.midi.parser.MidiFile(midi_file)
            tokens = tokenizer.midi_to_tokens(midi)
            
            # For multi-track tokenizers, tokens is a dict
            if isinstance(tokens, dict):
                for track_tokens in tokens.values():
                    if track_tokens:
                        token_sequences.append(track_tokens)
            else:
                token_sequences.append(tokens)
                
        except Exception as e:
            errors += 1
            if errors <= 5:  # Print only first 5 errors
                print(f"Error processing {midi_file}: {e}")
    
    print(f"Processed {len(token_sequences)} token sequences with {errors} errors")
    
    # Learn vocabulary
    print(f"Learning vocabulary (size: {args.vocab_size})...")
    tokenizer.learn_bpe(token_sequences, vocab_size=args.vocab_size)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save_params(tokenizer_path)
    
    # Save additional info
    info = {
        "tokenizer_type": args.tokenizer_type,
        "vocab_size": args.vocab_size,
        "special_tokens": args.special_tokens.split(","),
        "midi_files_processed": len(midi_files),
        "token_sequences_processed": len(token_sequences),
        "errors": errors
    }
    
    with open(output_dir / "tokenizer_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Analyze token distribution
    print("Analyzing token distribution...")
    all_tokens = [token for seq in token_sequences for token in seq]
    token_counts = Counter(all_tokens)
    
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Unique tokens: {len(token_counts)}")
    print(f"Most common tokens: {token_counts.most_common(10)}")

if __name__ == "__main__":
    args = parse_args()
    train_tokenizer(args)