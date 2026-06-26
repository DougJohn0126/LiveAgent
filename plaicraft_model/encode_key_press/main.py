#!/usr/bin/env python3

import argparse
import os
import sys
import sqlite3
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import logging
import pickle

from .scripts.key_press_encoder import KeyPressAutoencoder
from .scripts.constants import id_to_index, id_to_name  # Ensure these exist and map key IDs to indices

CHECKPOINT_DIR = "encode_key_press/checkpoints/keyencoder_16_5_best_checkpoint.pt"

###############################################################################
#                               HELPER FUNCTIONS                              #
###############################################################################

def setup_logging(verbose):
    """Configure logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    return logger

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Encode KeyPress and Mouse Click Data for a Session")
    parser.add_argument('--session_db_path', type=str, required=True,
                        help="Path to the session's SQLite database file")
    parser.add_argument('--model_checkpoint', type=str, default="/ubc/cs/research/ubc_ml/plaicraft/plaicraft-data-preprocessing/encode_key_press/checkpoints/keyencoder_16_5_best_checkpoint.pt",
                        help="Path to the trained model checkpoint (e.g., best.pt)")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size for encoding")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to run encoding on ('cuda' or 'cpu')")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose logging")
    return parser.parse_args()

class KeyPressDataset(Dataset):
    """
    Dataset for Key Press and Mouse Click Encodings.

    Each item is a 100ms window with:
        - session_id: str
        - start_timestamp: int (ms)
        - end_timestamp: int (ms)
        - keyboard_events: list of dicts
        - mouse_click_events: list of dicts
    """
    def __init__(self, windows, id_to_index, window_duration_ms=100, original_seq_len=10):
        """
        Args:
            windows (list of dict): Each dict contains 'session_id', 'start_timestamp', 'end_timestamp',
                                     'keyboard_events', 'mouse_click_events'.
            id_to_index (dict): Mapping from key_id to index.
            window_duration_ms (int): Duration of each window in milliseconds.
            original_seq_len (int): Sequence length of the input to the model.
        """
        self.windows = windows
        self.id_to_index = id_to_index
        self.input_dim = len(id_to_index)
        self.seq_len = original_seq_len  # As per training
        self.window_duration_ms = window_duration_ms

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        keyboard_data = window['keyboard_events']  # List of keyboard event dicts
        mouse_click_data = window.get('mouse_click_events', [])  # List of mouse click event dicts
        # Initialize tensor
        frame_tensor = torch.zeros(self.input_dim, self.seq_len, dtype=torch.float32)
        # Assign keyboard key presses
        for event in keyboard_data:
            key_id = event["key_id"]

            if key_id not in self.id_to_index:
                continue

            index = self.id_to_index[key_id]
            start_time = event["start_time"]
            end_time = event["end_time"]

            # Adjust end_time for scroll events to ensure non-zero duration
            if id_to_name.get(key_id, "").startswith("scroll_"):
                end_time = min(end_time + (1.0 / self.seq_len), 1.0)  # Add one bin's width

            # Calculate which time bins the key is active
            for t in range(self.seq_len):
                bin_start = t * (1.0 / self.seq_len)
                bin_end = (t + 1) * (1.0 / self.seq_len)
                if start_time < bin_end and end_time > bin_start:
                    frame_tensor[index, t] = 1.0

        # Assign mouse click and scroll events
        for event in mouse_click_data:
            mouse_key_type = event['mouse_key_type']  # e.g., 'mouse_left', 'scroll_up'
            start_time = event['start_time']  # Relative to window start, normalized [0,1)
            end_time = event['end_time']      # Relative to window start, normalized [0,1)

            # Adjust end_time for scroll events to ensure non-zero duration
            if mouse_key_type in ['scroll_up', 'scroll_down']:
                end_time = min(end_time + (1.0 / self.seq_len), 1.0)  # Add one bin's width

            # Calculate which time bins the mouse event is active
            for t in range(self.seq_len):
                bin_start = t * (1.0 / self.seq_len)
                bin_end   = (t + 1) * (1.0 / self.seq_len)
                if start_time < bin_end and end_time > bin_start:
                    mouse_index = self.id_to_index.get(mouse_key_type, None)
                    if mouse_index is not None:
                        frame_tensor[mouse_index, t] = 1.0

        return frame_tensor, {
            'session_id': window['session_id'],
            'start_timestamp': window['start_timestamp'],
            'end_timestamp': window['end_timestamp']
        }

def create_key_press_dataset(session_db, window_duration_ms=100, logger=None):
    """
    Read keypress and mouse click events from the database and prepare the dataset based on 100ms windows.

    Args:
        session_db_ (connection): the session's SQLite database.
        window_duration_ms (int): Duration of each window in milliseconds.

    Returns:
        KeyPressDataset: Prepared dataset.
    """
    con = session_db
    cur = con.cursor()

    # Fetch all unique session_ids from keyboard and mouse_click tables
    cur.execute("SELECT DISTINCT session_id FROM keyboard UNION SELECT DISTINCT session_id FROM mouse_click")
    session_ids = [row[0] for row in cur.fetchall()]

    all_windows = []

    logger.info(f"Processing count {len(session_ids)}")
    for session_id in session_ids:
        # Fetch session FPS and start_time
        cur.execute("""
            SELECT fps, start_time FROM session
            WHERE session_id = ?
        """, (session_id,))
        session_info = cur.fetchone()
        if not session_info:
            logging.warning(f"Session info not found for session_id {session_id}. Skipping.")
            continue
        fps, session_start_time = session_info
        if fps <= 0:
            logging.warning(f"Session {session_id} has non-positive FPS ({fps}). Skipping.")
            continue
        logger.info(f"Processing session_id: {session_id}")
        # Calculate total duration based on keyboard and mouse click events
        cur.execute("""
            SELECT MAX(end_timestamp) FROM keyboard
            WHERE session_id = ?
        """, (session_id,))
        max_end_keyboard = cur.fetchone()[0]
        logger.info(f"Processing session_id: {session_id}")
        cur.execute("""
            SELECT MAX(end_timestamp) FROM mouse_click
            WHERE session_id = ?
        """, (session_id,))
        max_end_mouse = cur.fetchone()[0]

        max_end_timestamp = max(filter(None, [max_end_keyboard, max_end_mouse]))
        if max_end_timestamp is None:
            logging.info(f"No keyboard or mouse click events found for session_id {session_id}. Skipping.")
            continue

        # Define windows
        num_windows = 50
        logger.info(max_end_timestamp)
        logger.info(f"Window count {num_windows}")
        for i in range(int(num_windows)):
            window_start = i * window_duration_ms + session_start_time
            window_end = window_start + window_duration_ms

            # Fetch keyboard events within this window
            cur.execute("""
                SELECT key_id, start_timestamp, end_timestamp FROM keyboard
                WHERE session_id = ?
                  AND start_timestamp < ?
                  AND end_timestamp > ?
            """, (session_id, window_end, window_start))
            keyboard_events = cur.fetchall()
            keyboard_event_list = []
            for event in keyboard_events:
                key_id, event_start, event_end = event
                # Adjust timestamps relative to window start and normalize to [0,1)
                rel_start = (event_start - window_start) / window_duration_ms
                rel_end = (event_end - window_start) / window_duration_ms
                rel_start = max(0.0, rel_start)
                rel_end = min(1.0, rel_end)
                keyboard_event_list.append({
                    'key_id': key_id,
                    'start_time': rel_start,
                    'end_time': rel_end
                })

            # Fetch mouse_click events within this window
            cur.execute("""
                SELECT mouse_key_type, start_timestamp, end_timestamp FROM mouse_click
                WHERE session_id = ?
                  AND start_timestamp < ?
                  AND end_timestamp > ?
            """, (session_id, window_end, window_start))
            mouse_clicks = cur.fetchall()
            mouse_click_event_list = []
            for event in mouse_clicks:
                mouse_key_type, event_start, event_end = event
                # Adjust timestamps relative to window start and normalize to [0,1)
                rel_start = (event_start - window_start) / window_duration_ms
                rel_end = (event_end - window_start) / window_duration_ms
                rel_start = max(0.0, rel_start)
                rel_end = min(1.0, rel_end)
                mouse_click_event_list.append({
                    'mouse_key_type': mouse_key_type,
                    'start_time': rel_start,
                    'end_time': rel_end
                })

            all_windows.append({
                'session_id': session_id,
                'start_timestamp': int(window_start),
                'end_timestamp': int(window_end),
                'keyboard_events': keyboard_event_list,
                'mouse_click_events': mouse_click_event_list
            })

    dataset = KeyPressDataset(all_windows, id_to_index, window_duration_ms=window_duration_ms)
    return dataset

def save_encodings_to_db(session_db, encodings):
    """
    Delete the 'key_press_encodings' table if it exists and create a new one.

    Args:
        session_db_path (str): Path to the session's SQLite database.
        encodings (list of tuples): Each tuple contains (session_id, start_timestamp, end_timestamp, encoding_blob).
    """
    con = session_db
    cur = con.cursor()
    try:
        logging.info("Dropping existing 'key_press_encodings' table if it exists...")
        cur.execute("DROP TABLE IF EXISTS key_press_encodings")
        con.commit()
        
        # Ensure the table exists
        cur.execute("""
            CREATE TABLE key_press_encodings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                start_timestamp INTEGER,
                end_timestamp INTEGER,
                encoding BLOB
            )
        """)
        con.commit()

        # Insert encodings
        cur.executemany("""
            INSERT INTO key_press_encodings (session_id, start_timestamp, end_timestamp, encoding)
            VALUES (?, ?, ?, ?)
        """, encodings)
        con.commit()
    except Exception as e:
        logging.error(f"Failed to insert encodings into 'key_press_encodings': {e}")
        con.rollback()
        sys.exit(1)
    finally:
        con.close()

###############################################################################
#                                MAIN FUNCTION                               #
###############################################################################

def main(sqDB, batch_size, device, verbose):
    logger = setup_logging(verbose)
    torch.zeros(1).cuda()
    # Validate device
    if device not in ['cuda', 'cpu']:
        logger.error("Invalid device specified. Choose 'cuda' or 'cpu'.")
        sys.exit(1)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load the dataset with 100ms windows, including mouse_click_events
    logger.info("Loading key press and mouse click data with 100ms windows...")
    dataset = create_key_press_dataset(sqDB, window_duration_ms=100, logger= logger)
    logger.info(f"Total windows to encode: {len(dataset)}")

    if len(dataset) == 0:
        logger.warning("No windows found for encoding. Exiting.")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"PyTorch Version: {torch.__version__}")

    # Initialize DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size= batch_size,
        shuffle=False,
        num_workers=0,  # Adjust based on your system
        pin_memory=True 
    )

    # Initialize the model
    logger.info("Initializing the KeyPressAutoencoder model...")
    autoencoder = KeyPressAutoencoder(
        input_dim=79,
        latent_dim=16,
        latent_seq_len=5,
        original_seq_len=10,
        num_gru_layers=2,
        conv_dropout=0.1,
        gru_dropout=0.1
    ).to(device)

    # Load the trained checkpoint
    logger.info(f"Loading model checkpoint from {CHECKPOINT_DIR}...")
    if not os.path.isfile(CHECKPOINT_DIR):
        logger.error(f"Model checkpoint not found: {CHECKPOINT_DIR}")
        sys.exit(1)

    try:
        checkpoint = torch.load(CHECKPOINT_DIR, map_location=device)
        autoencoder.load_state_dict(checkpoint)
        autoencoder.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model checkpoint: {e}")
        sys.exit(1)

    # Prepare to collect encodings
    all_encodings = []

    # Identify mouse action indices
    mouse_actions = ['mouse_left', 'mouse_right', 'scroll_up', 'scroll_down']
    mouse_indices = [id_to_index[ma] for ma in mouse_actions if ma in id_to_index]
    if not mouse_indices:
        logger.warning("No mouse action indices found in id_to_index mapping.")
    else:
        logger.info(f"Mouse action indices: {mouse_indices}")

    # Initialize a dictionary to count windows with mouse events per session
    session_mouse_window_counts = {}

    # Encode in batches
    logger.info("Starting encoding process...")
    with torch.no_grad():
        for batch_idx, (batch_data, batch_info) in enumerate(data_loader):
            batch_data = batch_data.to(device)  # Shape: (batch_size, input_dim, original_seq_len)
            z = autoencoder.encoder(batch_data)  # Shape: (batch_size, latent_dim, latent_seq_len)
            # Optionally, apply further processing on z if needed
            # For storage, serialize z as a BLOB
            z_cpu = z.cpu()
            batch_size_current = z_cpu.size(0)
            for i in range(batch_size_current):
                try:
                    # Ensure correct types
                    session_id = str(batch_info['session_id'][i])
                    start_timestamp = int(batch_info['start_timestamp'][i])
                    end_timestamp = int(batch_info['end_timestamp'][i])

                    # Initialize count for session if not already
                    if session_id not in session_mouse_window_counts:
                        session_mouse_window_counts[session_id] = 0

                    # Check if any mouse indices have non-zero values in this window
                    if mouse_indices:
                        # Extract the mouse part of the tensor for this window
                        mouse_data = batch_data[i, mouse_indices, :]  # Shape: (num_mouse_actions, seq_len)
                        if torch.any(mouse_data > 0):
                            session_mouse_window_counts[session_id] += 1

                    # Ensure encoding_blob is bytes
                    encoding_np = z_cpu[i].numpy().astype(np.float32)
                    encoding_blob = pickle.dumps(encoding_np)
                    all_encodings.append((session_id, start_timestamp, end_timestamp, encoding_blob))
                except Exception as e:
                    logger.error(f"Failed to process encoding for index {i} in batch {batch_idx}: {e}")
                    continue
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Encoded { (batch_idx + 1) * batch_size } windows...")

    # Save all encodings to the database
    logger.info(f"Saving {len(all_encodings)} encodings to the database...")
    save_encodings_to_db(sqDB, all_encodings)
    logger.info("Encodings saved successfully.")

    # Print the counts of windows with mouse events per session
    logger.info("Counting windows with non-zero mouse events per session...")
    for session_id, count in session_mouse_window_counts.items():
        logger.info(f"Session ID: {session_id} - Windows with mouse events: {count}")

    logger.info("Key press and mouse click encoding completed successfully.")

    return all_encodings  # Optionally return encodings for further use

if __name__ == "__main__":
    main()
