#!/usr/bin/env python3

import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import math
import traceback
import pickle
import logging
import numpy as np
from collections import defaultdict
import h5py 
from data.data_classes import FullData
from utils.constants import (
    AUDIO_FEATURE_DIM,
    AUDIO_TOKEN_FPS,
    AUDIO_TOKENS_PER_UNIT,
    KEYBOARD_TOKENS_PER_UNIT,
    KEYBOARD_TOKENS_PER_VIDEO_FRAME,
    MOUSE_TOKENS_PER_UNIT,
    MOUSE_TOKENS_PER_VIDEO_FRAME,
    VIDEO_FPS,
    VIDEO_FRAMES_PER_UNIT,
    VIDEO_LATENT_SHAPE,
)

class MapStyleDataset(Dataset):
    LATENT_FPS = VIDEO_FPS  # Video latent frames per second
    VIDEO_LATENTS_PER_BATCH = 100  # Number of video latents per stored batch file
    AUDIO_TOKEN_FRAME_RATE = AUDIO_TOKEN_FPS  # Audio tokens per second
    USE_FP16 = False
    BINS_PER_FRAME = MOUSE_TOKENS_PER_VIDEO_FRAME  # Number of bins per frame for mouse movement
    CLIP_MOUSE_DX = (-150.0, 150.0)
    CLIP_MOUSE_DY = (-100.0, 100.0)
    KEY_PRESS_ENC_PER_FRAME = KEYBOARD_TOKENS_PER_VIDEO_FRAME

    def __init__(self, dataset_path, modalities, window_length_frames=1, hop_length_frames=None, player_names=None, global_database_path=None):
        """
        Initialize the PlaicraftMapDataset.

        Args:
            dataset_path (str): Path to the dataset.
            modalities (list): List of modalities to load (e.g., ["video", "audio_speak", "audio_hear", "key_press", "mouse_movement"]).
            window_length_frames (int): Number of frames in each window.
            hop_length_frames (int, optional): Number of frames to move the window at each step. Defaults to window_length_frames (no overlap).
            player_names (list of str, optional): List of player names whose data to load.
            global_database_path (str): Path to the global metadata database
        """
        assert isinstance(window_length_frames, int) and window_length_frames > 0, \
            f"window_length_frames must be a positive integer, but got {window_length_frames}."

        if hop_length_frames is None:
            hop_length_frames = window_length_frames
        else:
            assert isinstance(hop_length_frames, int) and hop_length_frames > 0, \
                f"hop_length_frames must be a positive integer, but got {hop_length_frames}."

        if player_names is not None:
            player_names = list(player_names)

        self.dataset_path = Path(dataset_path)
        self.modalities = sorted(set(modalities))
        # window_length_frames represents the number of data units (pairs of frames) desired
        # Internally we need 2x frames since we group them into pairs
        self.window_length_frames = window_length_frames
        self.hop_length_frames = hop_length_frames
        self.player_names = player_names
        
        # Store the actual frame count needed (2 frames per unit)
        self._actual_window_frames = self.window_length_frames * VIDEO_FRAMES_PER_UNIT
        self._actual_hop_frames = self.hop_length_frames * VIDEO_FRAMES_PER_UNIT

        if global_database_path is None:
            self.global_db_path = self.dataset_path / "global_database.db"
        else:
            self.global_db_path = global_database_path
        self.connection = None
        self.session_windows = []

        self._init_dataset()

    def _init_dataset(self):
        print("trying to open global database: ", self.global_db_path)
        self.connection = sqlite3.connect(f'file:{self.global_db_path}?mode=ro', uri=True)
        self._load_sessions()
        self._build_index()

    def _load_sessions(self):
        cur = self.connection.cursor()

        base_conditions = []
        for m in self.modalities:
            if m == "video":
                base_conditions.append("video = 1")
            elif m == "audio_speak":
                base_conditions.append("audio_in = 1")
            elif m == "audio_hear":
                base_conditions.append("audio_out = 1")
            elif m == "key_press":
                base_conditions.append("keyboard = 1")
            elif m == "mouse_movement":
                base_conditions.append("mouse = 1")
        modalities_conditions = " AND ".join(base_conditions) if base_conditions else "1=1"

        params = []
        if self.player_names:
            placeholders = ','.join('?' * len(self.player_names))
            player_condition = f" AND player_name IN ({placeholders})"
            params.extend(self.player_names)
        else:
            player_condition = ""

        query = f"""
            SELECT session_id, player_name, start_time, frame_count, fps
            FROM session_metadata
            WHERE {modalities_conditions}{player_condition}
            ORDER BY player_name, start_time
        """
        try:
            cur.execute(query, params)
            session_rows = cur.fetchall()
            self.sessions = []
            for row in session_rows:
                session_id, player_name, start_time, frame_count, fps = row
                latent_frame_count = int(frame_count * (self.LATENT_FPS / fps))
                self.sessions.append({
                    "session_id": session_id,
                    "player_name": player_name,
                    "start_time": start_time,
                    "latent_frame_count": latent_frame_count
                })
            logging.info(f"Loaded {len(self.sessions)} sessions for player(s): {self.player_names}")
        except sqlite3.Error as e:
            logging.error(f"Error querying session_metadata: {e}")
            self.sessions = []

    def _build_index(self):
        """Index windows strictly within a single session (no cross-session spans)."""
        self.session_windows = []

        # group sessions by player (not strictly required, but keeps output similar)
        player_sessions = defaultdict(list)
        for s in self.sessions:
            player_sessions[s["player_name"]].append(s)

        # build windows inside each individual session
        for player, sessions in player_sessions.items():
            # keep chronological order for reproducibility
            sessions = sorted(sessions, key=lambda x: x["start_time"])

            for sess in sessions:
                L = int(sess["latent_frame_count"])
                if L < self._actual_window_frames:
                    continue  # session too short for one window

                win_start = 0
                while win_start + self._actual_window_frames <= L:
                    win_end = win_start + self._actual_window_frames
                    # one session per window; start/end are **session-relative**
                    self.session_windows.append({
                        "player_name": player,
                        "window_start_frame": win_start,
                        "window_end_frame": win_end,
                        "covered_sessions": [{
                            "session_id": sess["session_id"],
                            "start_frame": win_start,
                            "end_frame": win_end,
                        }],
                    })
                    win_start += self._actual_hop_frames

        self.total_windows = len(self.session_windows)
        logging.info(f"Total windows (single-session only): {self.total_windows}")

        if not self.total_windows:
            player_info = f" for players {self.player_names}" if self.player_names else ""
            raise ValueError(f"No valid data found{player_info} or the specified modalities.")

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(f"Index {idx} out of range.")

        window_info = self.session_windows[idx]
        player_name = window_info["player_name"]
        window_start_frame = window_info["window_start_frame"]
        window_end_frame = window_info["window_end_frame"]
        covered_sessions = window_info["covered_sessions"]

        # ---------- payload containers ----------
        data = {
            "metadata": [],
            "video": [],
            "audio_speak": [],
            "audio_hear": [],
            "key_press": [],
            "mouse_movement": [],
        }
        if "audio_speak" in self.modalities:
            data["transcript_speak"] = []
        if "audio_hear" in self.modalities:
            data["transcript_hear"] = []

        concat_buf = {m: [] for m in self.modalities if m not in {"key_press", "mouse_movement"}}

        for sess in covered_sessions:
            session_id  = sess["session_id"]
            start_frame = sess["start_frame"]
            end_frame   = sess["end_frame"]

            session_info = self._load_session_info(session_id)

            frame_length_ms = 1000 // self.LATENT_FPS
            sess_start_ts = start_frame * frame_length_ms
            sess_end_ts   = end_frame   * frame_length_ms

            data["metadata"].append({
                "player_id": session_info["player_id"],
                "player_name": session_info["player_name"],
                "player_email": session_info.get("player_email"),
                "player_gender": session_info.get("player_gender"),
                "player_skill_level": session_info.get("player_skill_level"),
                "session_id": session_id,
                "session_start_timestamp": session_info["start_time"],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "window_length_frames": end_frame - start_frame,
            })

            if "video" in self.modalities and session_info["modality_flags"]["video"]:
                v = self._load_video_encodings(
                        session_info["paths"]["video_encodings"],
                        start_frame, end_frame, session_id)
                concat_buf["video"].append(v)

            if "audio_speak" in self.modalities and session_info["modality_flags"]["audio_in"]:
                a_in = self._load_audio_encodings(
                        session_info["paths"]["audio_in_encodings"],
                        sess_start_ts, sess_end_ts)
                concat_buf["audio_speak"].append(a_in)
                data["transcript_speak"].extend(
                        self._load_transcripts(session_info["paths"]["db"],
                                               session_id, "transcript_in",
                                               sess_start_ts, sess_end_ts))

            if "audio_hear" in self.modalities and session_info["modality_flags"]["audio_out"]:
                a_out = self._load_audio_encodings(
                        session_info["paths"]["audio_out_encodings"],
                        sess_start_ts, sess_end_ts)
                concat_buf["audio_hear"].append(a_out)
                data["transcript_hear"].extend(
                        self._load_transcripts(session_info["paths"]["db"],
                                               session_id, "transcript_out",
                                               sess_start_ts, sess_end_ts))

            should_load_mouse = "mouse_movement" in self.modalities and session_info["modality_flags"].get("mouse")
            should_load_key = "key_press" in self.modalities and session_info["modality_flags"].get("keyboard")

            if should_load_mouse or should_load_key:
                sub_len = end_frame - start_frame
                mouse_tensor = None
                key_tensor = None
                try:
                    con = sqlite3.connect(f'file:{session_info["paths"]["db"]}?mode=ro', uri=True)
                    if should_load_mouse:
                        loaded_mouse = self._load_mouse_movement(
                            con,
                            session_id,
                            sess_start_ts,
                            sess_end_ts,
                            sub_len,
                        )
                        if torch.any(loaded_mouse != 0):
                            mouse_tensor = loaded_mouse
                    if should_load_key:
                        key_tensor = self._load_key_press_encoding(
                            con,
                            session_id,
                            sess_start_ts,
                            sess_end_ts,
                            sub_len,
                        )
                    con.close()
                except Exception as e:
                    logging.error(f"Error loading action data for session_id {session_id}: {e}")

                if "mouse_movement" in self.modalities:
                    if mouse_tensor is not None:
                        data["mouse_movement"].append(mouse_tensor)
                    else:
                        zeros = torch.zeros((2, self.BINS_PER_FRAME * sub_len), dtype=torch.float32)
                        data["mouse_movement"].append(zeros)

                if "key_press" in self.modalities:
                    if key_tensor is not None:
                        data["key_press"].append(key_tensor)
                    else:
                        zeros = torch.zeros((16, self.KEY_PRESS_ENC_PER_FRAME * sub_len), dtype=torch.float32)
                        data["key_press"].append(zeros)

        for mod in ["video", "audio_speak", "audio_hear"]:
            if mod in self.modalities:
                if concat_buf.get(mod):
                    if mod == "video":
                        data[mod] = torch.cat(concat_buf[mod], dim=0)
                    else:
                        data[mod] = torch.cat(concat_buf[mod], dim=1)
                else:
                    if mod == "video":
                        data[mod] = torch.zeros((self._actual_window_frames, *VIDEO_LATENT_SHAPE),
                                                dtype=torch.float16 if self.USE_FP16 else torch.float32)
                    else:
                        tok = math.ceil((self._actual_window_frames / self.LATENT_FPS) *
                                        self.AUDIO_TOKEN_FRAME_RATE)
                        data[mod] = torch.zeros((AUDIO_FEATURE_DIM, tok),
                                                dtype=torch.float16 if self.USE_FP16 else torch.float32)
            else:
                data[mod] = None

        for subk, bins, dim in (("mouse_movement", self.BINS_PER_FRAME, 2),
                                ("key_press", self.KEY_PRESS_ENC_PER_FRAME, 16)):
            if subk in self.modalities:
                if data[subk]:
                    data[subk] = torch.cat(data[subk], dim=1)
                else:
                    data[subk] = torch.zeros((dim, bins * self._actual_window_frames), dtype=torch.float32)
            else:
                data[subk] = None
        return FullData(batch=data)
    
    def _get_connection(self):
        """Lazy initialization of the SQLite connection.
           This ensures that each worker (after forking) creates its own connection."""
        if self.connection is None:
            self.connection = sqlite3.connect(f'file:{self.global_db_path}?mode=ro', uri=True)
        return self.connection

    def _load_session_info(self, session_id):
        cur = self._get_connection().cursor()
        cur.execute("""
            SELECT session_id, video, audio_in, audio_out, mouse, keyboard, fps, frame_count, start_time, player_email, player_name, player_id, player_gender, player_skill_level
            FROM session_metadata WHERE session_id = ?
        """, (session_id,))
        session_info = cur.fetchone()
        if not session_info:
            raise ValueError(f"No session info found for session ID: {session_id}")

        (sid, vid, aud_in, aud_out, m, kb, fps, fc, st, p_email, p_name, player_id, player_gender, player_skill_level) = session_info
        session_folder = self.dataset_path / p_email / sid
        modality_flags = {
            "video": vid,
            "audio_in": aud_in,
            "audio_out": aud_out,
            "mouse": m,
            "keyboard": kb
        }
        return {
            "session_id": sid,
            "player_name": p_name,
            "player_id": player_id,
            "player_gender": player_gender,
            "player_skill_level": player_skill_level,
            "player_email": p_email,
            "fps": fps,
            "frame_count": fc,
            "start_time": st,
            "modality_flags": modality_flags,
            "paths": {
                # Use HDF5 video encodings to match SSD-cache dataset backend
                "video_encodings": session_folder / "encoded_video_hdf5" / f"{sid}_encoded_video.hdf5",
                "audio_in_encodings": session_folder / "encoded_audio_continuous" / f"{sid}_encoded_audio_in.hdf5",
                "audio_out_encodings": session_folder / "encoded_audio_continuous" / f"{sid}_encoded_audio_out.hdf5",
                "db": session_folder / f"{sid}.db"
            }
        }

    def _load_transcripts(self, db_path, session_id, table_name, start_timestamp, end_timestamp):
        try:
            con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
            cur = con.cursor()
            cur.execute(f"""
                SELECT word, start_time, end_time
                FROM {table_name}
                WHERE session_id=?
                  AND (start_time >= ? AND end_time <= ?)
            """, (session_id, start_timestamp, end_timestamp))
            transcripts = cur.fetchall()
            con.close()
            return transcripts
        except:
            return []

    def _load_video_encodings(self, encodings_path, start_frame, end_frame, session_id):
        """Load video latents for [start_frame:end_frame) strictly from HDF5."""
        encodings_path = Path(encodings_path)
        if not encodings_path.exists() or encodings_path.suffix.lower() != ".hdf5":
            import os, grp
            print(f"--- DEBUG INFO ---")
            print(f"My UID: {os.getuid()}")
            print(f"My GID: {os.getgid()}")
            try:
                print(f"My Groups: {os.getgroups()}")
                # print(f"Group names: {[grp.getgrgid(g).gr_name for g in os.getgroups()]}")
            except Exception as e:
                print(f"Could not list groups: {e}")
            print(f"------------------")
            raise FileNotFoundError(
                f"Video HDF5 file not found for session {session_id} at {encodings_path}"
            )

        try:
            with h5py.File(encodings_path, "r") as f:
                latents_np = f['latents'][start_frame:end_frame]
                min_vals_np = f['min_vals'][start_frame:end_frame]
                scales_np = f['scales'][start_frame:end_frame]
            sliced_latents = torch.from_numpy(latents_np)
            sliced_min_vals = torch.from_numpy(min_vals_np)
            sliced_scales = torch.from_numpy(scales_np)
            dequantized_latents = self._dequantize_from_int8(sliced_latents, sliced_min_vals, sliced_scales)
            return dequantized_latents
        except Exception as e:
            raise RuntimeError(f"Failed to load video HDF5 for session {session_id}: {e}")

    def _dequantize_from_int8(self, quantized_tensor, min_val, scale):
        while min_val.dim() < quantized_tensor.dim():
            min_val = min_val.unsqueeze(-1)
        while scale.dim() < quantized_tensor.dim():
            scale = scale.unsqueeze(-1)
        dequantized = quantized_tensor.to(torch.float32) * scale + min_val
        return dequantized.half() if self.USE_FP16 else dequantized

    def _load_audio_encodings(self, audio_encodings_path, start_ms, end_ms):
        try:
            with h5py.File(audio_encodings_path, "r") as hf:
                tokens_dataset = hf["audio_latents"]
                duration_ms = end_ms - start_ms
                expected_audio_tokens = math.ceil((duration_ms / 1000) * self.AUDIO_TOKEN_FRAME_RATE)
                start_token = int(start_ms / 1000 * self.AUDIO_TOKEN_FRAME_RATE)
                end_token = start_token + expected_audio_tokens

                slice_np = tokens_dataset[0, :, start_token:end_token]
                tokens = torch.tensor(slice_np)
            if tokens.dim() == 3 and tokens.size(0) == 1:
                tokens = tokens.squeeze(0)
            elif tokens.dim() == 2:
                pass
            else:
                raise ValueError(f"Unexpected audio tokens shape: {tokens.shape}")

            current_tokens = tokens.shape[-1]
            if current_tokens > expected_audio_tokens:
                tokens = tokens[:, :expected_audio_tokens]
            if current_tokens < expected_audio_tokens:
                pad_size = expected_audio_tokens - current_tokens
                pad = torch.zeros((128, pad_size), dtype=tokens.dtype)
                tokens = torch.cat([tokens, pad], dim=-1)
            return tokens

        except Exception as e:
            logging.error(f"Error loading audio encodings from {audio_encodings_path}: {e}")
            return torch.zeros((128, 1), dtype=torch.float32)

    def _load_mouse_movement(self, con, session_id, start_timestamp, end_timestamp, sub_session_frames):
        try:
            cur = con.cursor()
            cur.execute("""
                SELECT timestamp, mouseDX, mouseDY
                FROM mouse_movement
                WHERE session_id=?
                AND timestamp >= ?
                AND timestamp < ?
                ORDER BY timestamp ASC
            """, (session_id, start_timestamp, end_timestamp))
            rows = cur.fetchall()

            if not rows:
                return torch.zeros((2, self.BINS_PER_FRAME * sub_session_frames), dtype=torch.float32)

            events = np.array(rows, dtype=np.float32)

            frame_length_ms = 1000.0 / self.LATENT_FPS
            dt = events[:, 0] - start_timestamp

            frame_indices = np.floor(dt / frame_length_ms).astype(np.int32)
            frame_indices = np.clip(frame_indices, 0, sub_session_frames - 1)

            dt_in_frame = dt - frame_indices * frame_length_ms
            bin_width = frame_length_ms / self.BINS_PER_FRAME
            bin_indices = np.floor(dt_in_frame / bin_width).astype(np.int32)
            bin_indices = np.clip(bin_indices, 0, self.BINS_PER_FRAME - 1)

            overall_bin_indices = frame_indices * self.BINS_PER_FRAME + bin_indices
            total_bins = sub_session_frames * self.BINS_PER_FRAME

            binned_dx = np.bincount(overall_bin_indices, weights=events[:, 1], minlength=total_bins)
            binned_dy = np.bincount(overall_bin_indices, weights=events[:, 2], minlength=total_bins)

            binned_dx = binned_dx.reshape((sub_session_frames, self.BINS_PER_FRAME))
            binned_dy = binned_dy.reshape((sub_session_frames, self.BINS_PER_FRAME))

            binned_dx = np.where((binned_dx < self.CLIP_MOUSE_DX[0]) | (binned_dx > self.CLIP_MOUSE_DX[1]),
                                 0.0, binned_dx)
            binned_dy = np.where((binned_dy < self.CLIP_MOUSE_DY[0]) | (binned_dy > self.CLIP_MOUSE_DY[1]),
                                 0.0, binned_dy)

            binned_data = np.concatenate([binned_dx.reshape(1, -1), binned_dy.reshape(1, -1)], axis=0)

            return torch.tensor(binned_data, dtype=torch.float32)
        except Exception as e:
            logging.error(f"Error loading mouse movement: {e}")
            return torch.zeros((2, self.BINS_PER_FRAME * sub_session_frames), dtype=torch.float32)

    def _load_key_press_encoding(self, con, session_id, start_timestamp, end_timestamp, sub_session_frames):
        try:
            cur = con.cursor()
            frame_length_ms = 1000 / self.LATENT_FPS
            cur.execute("""
                SELECT start_timestamp, end_timestamp, encoding
                FROM key_press_encodings
                WHERE session_id=?
                AND end_timestamp >= ?
                AND start_timestamp < ?
                ORDER BY start_timestamp
            """, (session_id, start_timestamp, end_timestamp))
            rows = cur.fetchall()

            enc_map = {}
            for (st, et, enc_blob) in rows:
                enc_map[(st, et)] = enc_blob

            all_encs = []
            for frame_idx in range(sub_session_frames):
                f_start = int(start_timestamp + frame_idx * frame_length_ms)
                f_end = int(f_start + frame_length_ms)
                encoding_blob = enc_map.get((f_start, f_end), None)
                if encoding_blob is not None:
                    encoding_array = pickle.loads(encoding_blob)
                    enc_tensor = torch.tensor(encoding_array, dtype=torch.float32)
                else:
                    enc_tensor = torch.zeros((16, self.KEY_PRESS_ENC_PER_FRAME), dtype=torch.float32)
                all_encs.append(enc_tensor)
            if not all_encs:
                return None
            return torch.cat(all_encs, dim=1)
        except Exception as e:
            logging.error(f"Error loading key_press encoding: {e}")
            return None

    def collate_fn(self, batch):
        """Concatenates multiple windows by stacking them along the batch dimension.
        
        Takes N items from DataLoader and horizontally concatenates their modalities.
        Each window is reshaped so frames/tokens are grouped, then all windows are stacked.
        Returns a single FullData batch with batch_size=N.
        """
        collated_batch = {
            "metadata": [],
            "video": [],
            "audio_speak": [],
            "audio_hear": [],
            "mouse_movement": [],
            "key_press": [],
        }
        if "audio_speak" in self.modalities:
            collated_batch["transcript_speak"] = []
        if "audio_hear" in self.modalities:
            collated_batch["transcript_hear"] = []

        for item in batch:
            # Convert FullData to dict for processing
            item_dict = item.to_dict() if isinstance(item, FullData) else item
            collated_batch["metadata"].append(item_dict["metadata"])

            if "video" in self.modalities:
                collated_batch["video"].append(
                    item_dict["video"].reshape(-1, VIDEO_FRAMES_PER_UNIT, *item_dict["video"].shape[1:])
                )

            if "audio_speak" in self.modalities:
                collated_batch["audio_speak"].append(
                    item_dict["audio_speak"].permute(1, 0).reshape(
                        -1, AUDIO_TOKENS_PER_UNIT, item_dict["audio_speak"].shape[0]
                    )
                )
                collated_batch["transcript_speak"].append(item_dict["transcript_speak"])

            if "audio_hear" in self.modalities:
                collated_batch["audio_hear"].append(
                    item_dict["audio_hear"].permute(1, 0).reshape(
                        -1, AUDIO_TOKENS_PER_UNIT, item_dict["audio_hear"].shape[0]
                    )
                )
                collated_batch["transcript_hear"].append(item_dict["transcript_hear"])

            if "mouse_movement" in self.modalities and item_dict["mouse_movement"] is not None:
                mm = item_dict["mouse_movement"].permute(1, 0)
                mm = mm.reshape(
                    -1,
                    MOUSE_TOKENS_PER_UNIT,
                    item_dict["mouse_movement"].shape[0],
                )
                collated_batch["mouse_movement"].append(mm)

            if "key_press" in self.modalities and item_dict["key_press"] is not None:
                kp = item_dict["key_press"].permute(1, 0)
                kp = kp.reshape(
                    -1,
                    KEYBOARD_TOKENS_PER_UNIT,
                    item_dict["key_press"].shape[0],
                )
                collated_batch["key_press"].append(kp)

        # Stack only the modalities that are requested and actually have data
        for mod in ["video", "audio_speak", "audio_hear"]:
            if mod not in self.modalities:
                collated_batch[mod] = None
                continue

            if len(collated_batch[mod]) == 0:
                collated_batch[mod] = None
                continue

            collated_batch[mod] = torch.stack(collated_batch[mod], dim=0)

        for subk in ["mouse_movement", "key_press"]:
            if subk not in self.modalities:
                collated_batch[subk] = None
            elif collated_batch[subk]:
                collated_batch[subk] = torch.stack(collated_batch[subk], dim=0)
            else:
                collated_batch[subk] = None

        return FullData(batch=collated_batch)

    @staticmethod
    def add_command_line_options(argparser):
        argparser.add_argument('--dataset_path', type=str, default=None,
            help='Path to the dataset folder containing session data.'
        )
        argparser.add_argument('--modalities', type=str, nargs='+',
            default=["video", "audio_speak", "audio_hear", "key_press", "mouse_movement"],
            help='List of modalities to load (e.g., ["video", "audio_speak", "audio_hear", "key_press", "mouse_movement"]).'
        )
        argparser.add_argument('--window_length_frames', type=int, default=10,
            help='Number of frames per window (e.g., 1000 frames = 100000ms at 10Hz).'
        )
        argparser.add_argument('--hop_length_frames', type=int, default=None,
            help='Number of frames to move the window at each step. Defaults to window_length_frames (no overlap).'
        )
        argparser.add_argument('--player_names', type=str, nargs='*', default=None,
            help='List of player names whose data to load.'
        )
        argparser.add_argument(
            '--global_database_path', type=str, default=None,
            help='Path to the global database file.'
        )

    def __del__(self):
        if hasattr(self, "connection") and self.connection:
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["connection"] = None
        return state


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Plaicraft Map Dataset Loader (Action: mouse movement + keypress)")
    MapStyleDataset.add_command_line_options(parser)
    args = parser.parse_args()

    try:
        dataset = MapStyleDataset(
            dataset_path=args.dataset_path,
            modalities=args.modalities,
            window_length_frames=args.window_length_frames,
            hop_length_frames=args.hop_length_frames,
            player_names=args.player_names,
            global_database_path=args.global_database_path
        )

        data_loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=4,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            persistent_workers=True
        )

        num_batches_to_test = 10000
        for batch_idx, batch_data in enumerate(data_loader):
            # batch_data is a FullData tensorclass instance
            print(f"\n=== Batch {batch_idx} ===")
            batch_dict = batch_data.to_dict()
            
            print("Metadata in this batch:")
            # Use attribute access (.metadata) instead of dict access (["metadata"])
            print(batch_dict["metadata"])

            if "video" in dataset.modalities and batch_data.video is not None:
                print("Video tensor shape:", batch_data.video.shape)
                
            if "audio_speak" in dataset.modalities and batch_data.audio_speak is not None:
                print("Audio_speak shape:", batch_data.audio_speak.shape)
                
            if "audio_hear" in dataset.modalities and batch_data.audio_hear is not None:
                print("Audio_hear shape:", batch_data.audio_hear.shape)

            if "mouse_movement" in dataset.modalities and batch_data.mouse_movement is not None:
                print("Mouse movement shape:", batch_data.mouse_movement.shape)
            if "key_press" in dataset.modalities and batch_data.key_press is not None:
                print("Key press shape:", batch_data.key_press.shape)

            if "audio_speak" in dataset.modalities and batch_dict["transcript_speak"] is not None:
                print("Transcript_speak lengths:", [len(x) for x in batch_dict["transcript_speak"]])
                
            if "audio_hear" in dataset.modalities and batch_dict["transcript_hear"] is not None:
                print("Transcript_hear lengths:", [len(x) for x in batch_dict["transcript_hear"]])

            if batch_idx + 1 == num_batches_to_test:
                break

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()