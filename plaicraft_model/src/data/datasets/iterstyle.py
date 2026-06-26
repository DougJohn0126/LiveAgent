#!/usr/bin/env python3
import os
import math
import random
import logging
import sqlite3
import pickle
import time
import threading
import shutil  # For file copying

import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torch.distributed as dist

import h5py  # For HDF5 support

from einops import rearrange
from data.data_classes import FullData
from utils.constants import (
    AUDIO_TOKEN_FPS,
    KEYBOARD_TOKENS_PER_UNIT,
    KEYBOARD_TOKENS_PER_VIDEO_FRAME,
    MOUSE_TOKENS_PER_UNIT,
    MOUSE_TOKENS_PER_VIDEO_FRAME,
    VIDEO_FPS,
)

# Use a multiprocessing Manager to share our combined cache view among workers.
from multiprocessing import Manager


def pad_sequence_dims(sequences, dims, padding_value=0):
    """Pad tensors along the given dims so they share a common shape."""
    assert all(len(sequences[0].shape) == len(seq.shape) for seq in sequences[1:])
    assert isinstance(dims, tuple)

    max_shape = []
    for i in range(len(sequences[0].shape)):
        if i in dims:
            max_shape.append(max(seq.shape[i] for seq in sequences))
        else:
            static_dim = [seq.shape[i] for seq in sequences]
            assert min(static_dim) == max(static_dim)
            max_shape.append(sequences[0].shape[i])

    padded_tensor = torch.full([len(sequences)] + max_shape, padding_value,
                               dtype=sequences[0].dtype, device=sequences[0].device)
    for i, seq in enumerate(sequences):
        new_shape = tuple([i] + [slice(0, seq.shape[j]) for j in range(len(seq.shape))])
        padded_tensor[new_shape].copy_(seq)
    return padded_tensor


def resolve_rank():
    """Return distributed rank metadata with sensible defaults."""
    rank = 0
    world_size = 1
    local_rank = 0
    node_rank = 0

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", 0)))
        node_rank = int(os.getenv("NODE_RANK", os.getenv("SLURM_NODEID", 0)))
    else:
        rank = int(os.getenv("RANK", os.getenv("SLURM_PROCID", 0)))
        world_size = int(os.getenv("WORLD_SIZE", os.getenv("SLURM_NTASKS", 1)))
        local_rank = int(os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", 0)))
        node_rank = int(os.getenv("NODE_RANK", os.getenv("SLURM_NODEID", 0)))

    return dict(rank=rank, world_size=world_size, local_rank=local_rank, node_rank=node_rank)


class IterStyleDataset(Dataset):
    LATENT_FPS = VIDEO_FPS
    VIDEO_LATENTS_PER_BATCH = 100
    AUDIO_TOKEN_FRAME_RATE = AUDIO_TOKEN_FPS
    USE_FP16 = False

    BINS_PER_FRAME = MOUSE_TOKENS_PER_VIDEO_FRAME
    CLIP_MOUSE_DX = (-150.0, 150.0)
    CLIP_MOUSE_DY = (-100.0, 100.0)
    KEY_PRESS_ENC_PER_FRAME = KEYBOARD_TOKENS_PER_VIDEO_FRAME

    def __init__(self, args, seed=None):
        """
        This dataset continuously loads data from a pool of SSD cache buffers.
        At initialization, it synchronously copies several random chunks from HDD to SSD.
        Then a background thread continuously refreshes the pool by copying a new chunk.
        When a new chunk finishes copying, it immediately replaces the oldest buffer.
        The old buffer is then scheduled for deletion after a grace period.
        If --use_ssd_cache is not specified, the dataset will just read directly from the HDD.
        """
        self.args = args
        self.dataset_path = Path(args.dataset_path)
        self.original_dataset_path = Path(args.dataset_path)
        self.modalities = sorted(set(args.modalities))

        self.player_names = args.player_names

        # Set node_id and gpu_id as instance variables
        dist_info = resolve_rank()
        self.rank = dist_info["rank"]
        self.world_size = dist_info["world_size"]
        self.local_rank = dist_info["local_rank"]
        self.node_id = dist_info["node_rank"]
        self.gpu_id = self.local_rank
        print("[Dataloader] rank: ", self.rank, " world_size: ", self.world_size, " local_rank: ", self.local_rank, " node_id: ", self.node_id, " gpu_id: ", self.gpu_id)
        
        base_seed = 0 if seed is None else int(seed)
        self.rng = random.Random(base_seed + self.rank)

        if args.global_database_path is None:
            self.global_db_path = self.dataset_path / "global_database.db"
        else:
            self.global_db_path = Path(args.global_database_path)

        # Set the SSD cache directory from command-line options, defaulting to SLURM_TMPDIR or "/tmp"
        self.ssd_cache_dir = getattr(args, "ssd_cache_dir", os.environ.get("SLURM_TMPDIR", "/tmp"))

        self.connection = None
        self._init_dataset()
        self._compute_player_frame_distribution()
        self.player_count = len(self.player_names) if self.player_names else len(self.players_list)

        self.use_ssd_cache = getattr(args, "use_ssd_cache", False)
        self.chunk_size_gb = getattr(args, "chunk_size_gb", 2.0)
        if self.use_ssd_cache:
            logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: SSD cache enabled; setting up asynchronous multi-buffering SSD cache.")
            self.cache_pool_size = getattr(args, "cache_queue_size", 4)  # e.g. 4 buffers
            self.active_cache_pool = []  # List of cache info dicts
            self.cache_lock = threading.Lock()
            self.manager = Manager()
            self.shared_cache_info = self.manager.dict()
            self.shared_cache_info["cache_state"] = {
                "combined_player_infos": {}, 
                "combined_players_list": [],
                "combined_frames_list": [],
                "combined_total_frames": 0,
                "session_info_map": {}
            }
            self.ssd_cache_removal_grace_period = getattr(args, "ssd_cache_removal_grace_period", 60)
            self.refresh_interval = getattr(args, "ssd_cache_refresh_interval", 5)
            
            self.iterations_per_cache = getattr(args, "iterations_per_cache", 10_000)
            self.iterations_since_last_swap = 0

            # one-chunk look-ahead synchronisation
            self.pending_cache_info = None
            self.pending_ready_event = threading.Event()   # set by copier-thread
            self.swap_done_event   = threading.Event()     # set by monitor thread
            self.swap_done_event.set()                     # allow first copy to start

            # >>> NEW: process-shared batch counter <<<
            self.batch_counter = self.manager.Value('i', 0)          # starts at 0
            self.counter_lock  = self.manager.Lock()

            # start helper threads
            self._compute_chunks()
            self._load_initial_cache_pool()
            self._update_combined_cache_info()
            self._start_async_cache_thread()
            self._start_batch_monitor_thread()
        else:
            self._build_combined_cache_info_from_hdd()
            
    def _start_batch_monitor_thread(self):
        t = threading.Thread(target=self._batch_monitor_loop, daemon=True)
        t.start()
        self.batch_monitor_thread = t
        
    def _batch_monitor_loop(self):
        """
        Runs only in the *original* dataset object (main process).
        Swaps the cache whenever BOTH:
            1)   exactly `iterations_per_cache` batches have been delivered
            2)   the copier thread has a fresh chunk waiting
        The two events can arrive in any order.
        """
        while True:
            # ---------- wait until enough batches have been produced ----------
            while True:
                with self.counter_lock:
                    if self.batch_counter.value >= self.iterations_per_cache:
                        break
                time.sleep(0.05)          # light-weight poll

            # ---------- wait for the prefetch to finish ----------
            self.pending_ready_event.wait()   # blocks here if chunk not ready
            self.pending_ready_event.clear()

            # ---------- do the swap (atomic) ----------
            self._swap_in_pending_cache()

            # reset the counter for the next cycle
            with self.counter_lock:
                self.batch_counter.value = 0

            # allow copier thread to begin fetching the next chunk
            self.swap_done_event.set()
            logging.info(
                f"Node {self.node_id}, GPU {self.gpu_id}: "
                f"Cache swapped after {self.iterations_per_cache} batches."
            )


    # ------------- Initial Pool & Combined Cache View -------------
    def _load_initial_cache_pool(self):
        ssd_base = self.ssd_cache_dir
        for i in range(self.cache_pool_size):
            chunk = self.rng.choice(self.chunks)
            chunk_idx = self.chunks.index(chunk)
            target_path = Path(ssd_base) / "plaicraft_ssd_cache" / f"node_{self.node_id}_gpu_{self.gpu_id}" / f"chunk_{chunk_idx}_buffer_{int(time.time())}"
            logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Loading initial cache buffer {i+1}/{self.cache_pool_size} from chunk #{chunk_idx} at {target_path}")
            self._copy_chunk_to_ssd(chunk, target_path)
            cache_info = self._build_cache_info(chunk, target_path)
            self.active_cache_pool.append(cache_info)
            time.sleep(1)
        logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Initial active cache pool: {[c['chunk_idx'] for c in self.active_cache_pool]}")

    def _build_cache_info(self, chunk, target_path):
        chunk_idx = self.chunks.index(chunk)
        new_player_infos = {}
        new_player_sessions = {}
        for sess in chunk["sessions"]:
            player = sess["player"]
            sess_copy = sess.copy()
            sess_copy["ssd_cache_path"] = target_path
            if player not in new_player_infos:
                new_player_infos[player] = {"total_frames": 0, "session_ranges": []}
                new_player_sessions[player] = []
            start_offset = new_player_infos[player]["total_frames"]
            end_offset = start_offset + sess["latent_frame_count"]
            session_range = {
                "session_id": sess_copy["session_id"],
                "player_email": sess_copy["player_email"],
                "start_offset": start_offset,
                "end_offset": end_offset,
                "latent_frame_count": sess_copy["latent_frame_count"],
                "db_start_time": sess_copy["start_time"],
                "ssd_cache_path": target_path
            }
            session_range["player_gender"] = sess_copy.get("player_gender")
            session_range["player_skill_level"] = sess_copy.get("player_skill_level")
            new_player_infos[player]["session_ranges"].append(session_range)
            new_player_infos[player]["total_frames"] = end_offset
            new_player_sessions[player].append(sess_copy)
        info = {
            "ssd_cache_path": target_path,
            "player_infos": new_player_infos,
            "player_sessions": new_player_sessions,
            "chunk_idx": chunk_idx,
            "refcount": 0,
            "marked_for_removal": False,
        }
        logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Built cache info for chunk #{chunk_idx} at {target_path}")
        return info

    def _update_combined_cache_info(self):
        logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Starting combined cache info update")
        combined_player_infos = {}
        for cache in self.active_cache_pool:
            for player, info in cache["player_infos"].items():
                if player not in combined_player_infos:
                    combined_player_infos[player] = {"session_ranges": []}
                combined_player_infos[player]["session_ranges"].extend(info["session_ranges"])
        logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Aggregated player session ranges")

        combined_players_list = []
        combined_frames_list = []
        total_frames = 0
        new_session_info_map = {}
        for player, info in combined_player_infos.items():
            sorted_sessions = sorted(info["session_ranges"], key=lambda x: x["db_start_time"])
            global_offset = 0
            for session in sorted_sessions:
                session["global_start_offset"] = global_offset
                session["global_end_offset"] = global_offset + session["latent_frame_count"]
                global_offset += session["latent_frame_count"]
                new_session_info_map[session["session_id"]] = session
            combined_player_infos[player]["total_frames"] = global_offset
            combined_player_infos[player]["session_ranges"] = sorted_sessions
            combined_players_list.append(player)
            combined_frames_list.append(global_offset)
            total_frames += global_offset
        logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Computed global offsets and session map")

        self.combined_player_infos = combined_player_infos
        self.combined_players_list = combined_players_list
        self.combined_frames_list = combined_frames_list
        self.combined_total_frames = total_frames
        logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Updating shared cache info")

        new_state = {
            "combined_player_infos": combined_player_infos,
            "combined_players_list": combined_players_list,
            "combined_frames_list": combined_frames_list,
            "combined_total_frames": total_frames,
            "session_info_map": new_session_info_map
        }
        self.shared_cache_info["cache_state"] = new_state

        logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Shared cache info updated with total frames {total_frames}")
        logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Combined cache info update completed")

    def _start_async_cache_thread(self):
        self.cache_thread = threading.Thread(target=self._async_cache_refresh_loop, daemon=True)
        self.cache_thread.start()
        logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Started asynchronous cache refresh thread.")

    def _async_cache_refresh_loop(self):
        """
        Background copier:
            – keeps *at most one* chunk waiting in `self.pending_cache_info`
            – blocks until `swap_done_event` is set (i.e. the waiting chunk
            has been consumed) before it starts copying the next one
        """
        while True:
            # block until main thread tells us the previous pending chunk has
            # been consumed
            self.swap_done_event.wait()
            self.swap_done_event.clear()

            try:
                # choose and copy a new chunk
                ssd_base = self.ssd_cache_dir
                new_chunk    = self.rng.choice(self.chunks)
                new_idx      = self.chunks.index(new_chunk)
                new_target   = (Path(ssd_base) /
                                "plaicraft_ssd_cache" /
                                f"node_{self.node_id}_gpu_{self.gpu_id}" /
                                f"chunk_{new_idx}_buffer_{int(time.time())}")

                logging.info(
                    f"Node {self.node_id}, GPU {self.gpu_id}: "
                    f"Prefetching chunk #{new_idx} → {new_target}"
                )
                self._copy_chunk_to_ssd(new_chunk, new_target)
                new_info = self._build_cache_info(new_chunk, new_target)

                with self.cache_lock:
                    self.pending_cache_info = new_info

                self.pending_ready_event.set()   # signal availability

            except Exception as e:
                logging.error(
                    f"Node {self.node_id}, GPU {self.gpu_id}: "
                    f"Copier thread error: {e}"
                )
                # let main thread continue; try again next round
                self.pending_ready_event.clear()

            # small sleep just to be polite to the scheduler
            time.sleep(self.refresh_interval)

    
    def _swap_in_pending_cache(self):
        """
        Atomically replace the oldest cache entry with the pending one.
        Must be called with no locks held.
        """
        with self.cache_lock:
            old_cache = self.active_cache_pool.pop(0)
            new_cache = self.pending_cache_info
            self.active_cache_pool.append(new_cache)
            self.pending_cache_info = None
            self._update_combined_cache_info()
            
            logging.info(
                f"Node {self.node_id}, GPU {self.gpu_id}: "
                f"Active cache pool after swap: "
                f"{[c['chunk_idx'] for c in self.active_cache_pool]}"
            )

        # schedule deferred removal of the evicted chunk
        self._schedule_removal(old_cache)


    def _schedule_removal(self, cache_info):
        def delayed():
            with self.cache_lock:
                if cache_info["refcount"] == 0:
                    self._really_remove_chunk(cache_info)
                else:
                    # Mark for removal so that once refcount drops to 0, removal occurs.
                    cache_info["marked_for_removal"] = True
        t = threading.Timer(self.ssd_cache_removal_grace_period, delayed)
        t.daemon = True
        t.start()

    def _really_remove_chunk(self, cache_info):
        path = cache_info["ssd_cache_path"]
        chunk_idx = cache_info["chunk_idx"]
        try:
            shutil.rmtree(path)
            logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Removed old cache for chunk #{chunk_idx} at {path}")
        except Exception as e:
            logging.error(f"Node {self.node_id}, GPU {self.gpu_id}: Failed to remove old cache for chunk #{chunk_idx} at {path}: {e}")

    # ---------------- Standard Dataset Methods ----------------
    def _get_connection(self):
        if self.connection is None:
            self.connection = sqlite3.connect(f'file:{self.global_db_path}?mode=ro', uri=True)
        return self.connection

    def _init_dataset(self):
        tmp_conn = sqlite3.connect(f'file:{self.global_db_path}?mode=ro', uri=True)
        self.connection = tmp_conn
        self._load_sessions()
        tmp_conn.close()
        self.connection = None

    def _load_sessions(self):
        cur = self.connection.cursor()
        base_conditions = []
        for m in self.modalities:
            if m == "audio_speak":
                base_conditions.append("audio_in = 1")
            elif m == "audio_hear":
                base_conditions.append("audio_out = 1")
            elif m == "key_press":
                base_conditions.append("keyboard = 1")
            elif m == "mouse_movement":
                base_conditions.append("mouse = 1")
            else:
                base_conditions.append(f"{m} = 1")
        modalities_conditions = " AND ".join(base_conditions) if base_conditions else "1=1"
        params = []
        if self.player_names:
            placeholders = ','.join('?' * len(self.player_names))
            player_condition = f" AND player_name IN ({placeholders})"
            params.extend(self.player_names)
        else:
            player_condition = ""
        query = f"""
        SELECT session_id, player_id, player_name, player_email, player_gender, player_skill_level, start_time, frame_count, fps
        FROM session_metadata
        WHERE {modalities_conditions}{player_condition}
        ORDER BY player_name, start_time
        """
        try:
            cur.execute(query, params)
            session_rows = cur.fetchall()
            self.player_sessions = defaultdict(list)
            for row in session_rows:
                session_id, player_id, player_name, player_email, player_gender, player_skill_level, start_time, frame_count, fps = row
                latent_frame_count = int(frame_count * (self.LATENT_FPS / fps))
                self.player_sessions[player_name].append({
                    "session_id": session_id,
                    "player_id": player_id,
                    "player_email": player_email,
                    "player_gender": player_gender,
                    "player_skill_level": player_skill_level,
                    "start_time": start_time,
                    "latent_frame_count": latent_frame_count
                })
            for p in self.player_sessions:
                self.player_sessions[p].sort(key=lambda x: x["start_time"])
            self.player_infos = {}
            for player, sessions in self.player_sessions.items():
                session_ranges = []
                cumulative_offset = 0
                for s in sessions:
                    start_offset = cumulative_offset
                    end_offset = start_offset + s["latent_frame_count"]
                    session_ranges.append({
                        "session_id": s["session_id"],
                        "player_email": s["player_email"],
                        "player_gender": s["player_gender"],
                        "player_skill_level": s["player_skill_level"],
                        "start_offset": start_offset,
                        "end_offset": end_offset,
                        "latent_frame_count": s["latent_frame_count"],
                        "db_start_time": s["start_time"]
                    })
                    cumulative_offset = end_offset
                self.player_infos[player] = {
                    "total_frames": cumulative_offset,
                    "session_ranges": session_ranges
                }
        except sqlite3.Error as e:
            logging.error(f"Node {self.node_id}, GPU {self.gpu_id}: Error querying session_metadata: {e}")
            self.player_sessions = {}
            self.player_infos = {}

    def _compute_player_frame_distribution(self):
        self.players_list = []
        self.frames_list = []
        self.total_frames_across_all_players = 0
        for p, info in self.player_infos.items():
            frames = info["total_frames"]
            if frames > 0:
                self.players_list.append(p)
                self.frames_list.append(frames)
                self.total_frames_across_all_players += frames
        if self.total_frames_across_all_players == 0:
            raise ValueError("No valid sessions or frames found.")

    def _compute_chunks(self):
        self.chunks = []
        frames_per_chunk = int(self.chunk_size_gb * 18000)
        all_sessions = []
        for player in sorted(self.player_sessions.keys()):
            for s in self.player_sessions[player]:
                all_sessions.append({
                    "session_id": s["session_id"],
                    "player": player,
                    "player_email": s["player_email"],
                    "latent_frame_count": s["latent_frame_count"],
                    "start_time": s["start_time"]
                })
        all_sessions.sort(key=lambda x: (x["player"], x["start_time"]))
        current_chunk = []
        current_total = 0
        for sess in all_sessions:
            if current_chunk and (current_total + sess["latent_frame_count"] > frames_per_chunk):
                self.chunks.append({"sessions": current_chunk, "total_frames": current_total})
                current_chunk = []
                current_total = 0
            current_chunk.append(sess)
            current_total += sess["latent_frame_count"]
        if current_chunk:
            self.chunks.append({"sessions": current_chunk, "total_frames": current_total})
        logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Computed {len(self.chunks)} SSD chunks.")

    def _fast_copy_dir(self, src, dst, extension):
        cmd = f"rsync -a --no-group --include='*/' --include='*{extension}' --exclude='*' {src}/ {dst}/"
        logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Executing rsync command: {cmd}")
        ret = os.system(cmd)
        if ret != 0:
            logging.warning(f"Node {self.node_id}, GPU {self.gpu_id}: rsync command returned non-zero status: {ret}")

    def _copy_session_files(self, src, dst):
        try:
            os.makedirs(dst, exist_ok=True)
            
            video_src = src / "encoded_video_hdf5"
            video_dst = dst / "encoded_video_hdf5"
            if video_src.exists():
                self._fast_copy_dir(video_src, video_dst, ".hdf5")
            else:
                logging.warning(f"Node {self.node_id}, GPU {self.gpu_id}: Video folder not found in {src}")
            audio_src = src / "encoded_audio_continuous"
            audio_dst = dst / "encoded_audio_continuous"
            if audio_src.exists():
                self._fast_copy_dir(audio_src, audio_dst, ".hdf5")
            else:
                logging.warning(f"Node {self.node_id}, GPU {self.gpu_id}: Audio folder not found in {src}")
            db_file = src / (src.name + ".db")
            if db_file.exists():
                shutil.copy2(db_file, dst)
            else:
                logging.warning(f"Node {self.node_id}, GPU {self.gpu_id}: Session DB file not found in {src}")
        except Exception as e:
            logging.error(f"Node {self.node_id}, GPU {self.gpu_id}: Error copying session files from {src} to {dst}: {e}")

    def _copy_chunk_to_ssd(self, chunk, target_path):
        if target_path.exists():
            logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: SSD cache directory {target_path} already exists. Skipping copy.")
            return
        try:
            os.makedirs(target_path, exist_ok=True)
            for sess in chunk["sessions"]:
                src = self.original_dataset_path / sess["player_email"] / sess["session_id"]
                dst = target_path / sess["player_email"] / sess["session_id"]
                if not src.exists():
                    logging.warning(f"Node {self.node_id}, GPU {self.gpu_id}: Source session folder {src} does not exist; skipping.")
                    continue
                os.makedirs(dst.parent, exist_ok=True)
                logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Copying session {sess['session_id']} from {src} to {dst}")
                self._copy_session_files(src, dst)
            logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Finished copying chunk to SSD cache at {target_path}")
        except Exception as e:
            logging.error(f"Node {self.node_id}, GPU {self.gpu_id}: Error copying chunk to SSD: {e}")

    def _load_session_info(self, session_id):
        if self.use_ssd_cache:
            try:
                cache_state = self.shared_cache_info.get("cache_state", {})
                session_info_map = cache_state.get("session_info_map", {})
                if session_id not in session_info_map:
                    logging.warning(f"Node {self.node_id}, GPU {self.gpu_id}: Session {session_id} missing in cache. Falling back to DB.")
                    return self._load_session_info_from_db(session_id)
                info = session_info_map[session_id]
                player_email = info["player_email"]
                sid = info["session_id"]
                session_folder = info["ssd_cache_path"] / player_email / sid
                # Increment reference count for the chunk before reading.
                cache_chunk = None
                with self.cache_lock:
                    for cache in self.active_cache_pool:
                        if cache["ssd_cache_path"] == info["ssd_cache_path"]:
                            cache["refcount"] += 1
                            cache_chunk = cache
                            break
                cur = self._get_connection().cursor()
                cur.execute("""
                SELECT session_id, player_id, video, audio_in, audio_out, mouse, keyboard, fps, frame_count, start_time, player_email, player_name, player_gender, player_skill_level
                FROM session_metadata WHERE session_id = ?
                """, (session_id,))
                session_info = cur.fetchone()
                if not session_info:
                    logging.warning(f"Node {self.node_id}, GPU {self.gpu_id}: Session {session_id} not found in DB via cache lookup. Falling back to DB.")
                    return self._load_session_info_from_db(session_id)
                (sid, player_id, vid, aud_in, aud_out, m, kb, fps, fc, st, p_email, p_name, p_gender, p_skill) = session_info
                modality_flags = {"video": vid, "audio_speak": aud_in, "audio_hear": aud_out, "mouse": m, "keyboard": kb}
                result = {
                    "session_id": sid,
                    "player_id": player_id,
                    "player_name": p_name,
                    "player_email": p_email,
                    "player_gender": p_gender,
                    "player_skill_level": p_skill,
                    "fps": fps,
                    "frame_count": fc,
                    "start_time": st,
                    "modality_flags": modality_flags,
                    "paths": {
                        "video_encodings": session_folder / "encoded_video_hdf5" / f"{sid}_encoded_video.hdf5",
                        "audio_in_encodings": session_folder / "encoded_audio_continuous" / f"{sid}_encoded_audio_in.hdf5",
                        "audio_out_encodings": session_folder / "encoded_audio_continuous" / f"{sid}_encoded_audio_out.hdf5",
                        "db": session_folder / f"{sid}.db"
                    }
                }
                # Decrement reference count after reading.
                if cache_chunk is not None:
                    with self.cache_lock:
                        cache_chunk["refcount"] -= 1
                        if cache_chunk["marked_for_removal"] and cache_chunk["refcount"] == 0:
                            self._really_remove_chunk(cache_chunk)
                return result
            except Exception as e:
                logging.error(f"Node {self.node_id}, GPU {self.gpu_id}: Error loading session info from SSD for session {session_id}: {e}. Falling back to DB.")
                return self._load_session_info_from_db(session_id)
        else:
            return self._load_session_info_from_db(session_id)

    def _load_session_info_from_db(self, session_id):
        logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Falling back to global DB for session {session_id}")
        conn = sqlite3.connect(f'file:{self.global_db_path}?mode=ro', uri=True)
        cur = conn.cursor()
        cur.execute("""
        SELECT session_id, player_id, video, audio_in, audio_out, mouse, keyboard, fps, frame_count, start_time, player_email, player_name, player_gender, player_skill_level
        FROM session_metadata WHERE session_id = ?
        """, (session_id,))
        session_info = cur.fetchone()
        conn.close()
        if not session_info:
            raise ValueError(f"No session info found in DB for session ID: {session_id}")
        (sid, player_id, vid, aud_in, aud_out, m, kb, fps, fc, st, p_email, p_name, p_gender, p_skill) = session_info
        modality_flags = {"video": vid, "audio_speak": aud_in, "audio_hear": aud_out, "mouse": m, "keyboard": kb}
        session_folder = self.original_dataset_path / p_email / sid
        return {
            "session_id": sid,
            "player_id": player_id,
            "player_name": p_name,
            "player_email": p_email,
            "player_gender": p_gender,
            "player_skill_level": p_skill,
            "fps": fps,
            "frame_count": fc,
            "start_time": st,
            "modality_flags": modality_flags,
            "paths": {
                "video_encodings": session_folder / "encoded_video_hdf5" / f"{sid}_encoded_video.hdf5",
                "audio_in_encodings": session_folder / "encoded_audio_continuous" / f"{sid}_encoded_audio_in.hdf5",
                "audio_out_encodings": session_folder / "encoded_audio_continuous" / f"{sid}_encoded_audio_out.hdf5",
                "db": session_folder / f"{sid}.db"
            }
        }
        
    def _path_source_label(self, p) -> str:
        """
        Return 'SSD' if the path is inside the configured ssd_cache_dir, otherwise 'HDD'.
        Falls back to 'unknown' if the path can't be resolved.
        """
        try:
            p = Path(p)
        except Exception:
            return "unknown"
        ssd_root = Path(getattr(self, "ssd_cache_dir", "/tmp"))
        try:
            return "SSD" if (p == ssd_root or ssd_root in p.parents) else "HDD"
        except Exception:
            return "unknown"

    def _log_load_failure(self, modality: str, path: Path, session_id: str, exc: Exception, context: str):
        """
        Emit a consistent, source-aware error line for any failed load attempt.
        context: 'primary' or 'fallback'
        """
        src = self._path_source_label(path)
        logging.error(
            f"Node {self.node_id}, GPU {self.gpu_id}: {context} load failed for {modality} "
            f"[session {session_id}] from {src} path {path}: {type(exc).__name__}: {exc}"
        )
        
    def _build_combined_cache_info_from_hdd(self):
        """
        Build the same combined_player_infos / players_list / frames_list structure
        that the SSD path uses, but from self.player_infos (HDD metadata only).
        This is called once in __init__ when SSD cache is disabled.
        """
        combined_player_infos = {}
        combined_players_list = []
        combined_frames_list = []
        total_frames = 0

        for player, info in self.player_infos.items():
            # session_ranges are already in temporal order from _load_sessions,
            # but we keep the sort to be explicit.
            sorted_sessions = sorted(info["session_ranges"], key=lambda x: x["db_start_time"])
            global_offset = 0
            for session in sorted_sessions:
                session["global_start_offset"] = global_offset
                session["global_end_offset"] = global_offset + session["latent_frame_count"]
                global_offset += session["latent_frame_count"]

            combined_player_infos[player] = {
                "total_frames": global_offset,
                "session_ranges": sorted_sessions,
            }
            combined_players_list.append(player)
            combined_frames_list.append(global_offset)
            total_frames += global_offset

        self.combined_player_infos_hdd = combined_player_infos
        self.combined_players_list_hdd = combined_players_list
        self.combined_frames_list_hdd = combined_frames_list
        self.combined_total_frames_hdd = total_frames

    def __getitem__(self, data):
        window_length_frames = data

        if self.use_ssd_cache:
            cache_state = self.shared_cache_info.get("cache_state", {
                "combined_player_infos": self.combined_player_infos,
                "combined_players_list": self.combined_players_list,
                "combined_frames_list": self.combined_frames_list,
                "combined_total_frames": self.combined_total_frames,
                "session_info_map": {}
            })
            shared_player_infos = cache_state["combined_player_infos"]
            shared_players_list = cache_state["combined_players_list"]
            shared_frames_list = cache_state["combined_frames_list"]
            total_frames = cache_state["combined_total_frames"]
        else:
            shared_player_infos = self.combined_player_infos_hdd
            shared_players_list = self.combined_players_list_hdd
            shared_frames_list = self.combined_frames_list_hdd
            total_frames = self.combined_total_frames_hdd

        # Choose player weighted by frames, ensuring enough frames for the window
        eligible_players = []
        eligible_total_frames = 0
        for p, frames in zip(shared_players_list, shared_frames_list):
            if shared_player_infos[p]["total_frames"] >= window_length_frames:
                eligible_players.append((p, frames))
                eligible_total_frames += frames
        if not eligible_players:
            raise ValueError(f"No player has enough frames for window {window_length_frames}.")

        r = self.rng.uniform(0, eligible_total_frames)
        running = 0
        chosen_player = None
        for p, frames in eligible_players:
            if r < (running + frames):
                chosen_player = p
                break
            running += frames
        if chosen_player is None:
            chosen_player = eligible_players[-1][0]

        player_info = shared_player_infos[chosen_player]
        valid_range = player_info["total_frames"] - window_length_frames
        if valid_range < 0:
            raise ValueError(f"Player {chosen_player} does not have enough frames for window {window_length_frames}.")
        start_frame_in_player = self.rng.randint(0, valid_range)
        end_frame_in_player = start_frame_in_player + window_length_frames

        covered_sessions = []
        for sess in player_info["session_ranges"]:
            if sess["global_end_offset"] <= start_frame_in_player:
                continue
            if sess["global_start_offset"] >= end_frame_in_player:
                break
            overlap_start = max(sess["global_start_offset"], start_frame_in_player)
            overlap_end = min(sess["global_end_offset"], end_frame_in_player)
            frame_start_in_session = overlap_start - sess["global_start_offset"]
            frame_end_in_session = overlap_end - sess["global_start_offset"]
            # For HDD mode, if 'ssd_cache_path' is missing, fall back to original dataset location.
            covered_sessions.append({
                "session_id": sess["session_id"],
                "player_email": sess["player_email"],
                "db_start_time": sess["db_start_time"],
                "start_frame_in_session": frame_start_in_session,
                "end_frame_in_session": frame_end_in_session,
                "ssd_cache_path": sess.get("ssd_cache_path", self.original_dataset_path / sess["player_email"] / sess["session_id"])
            })

        data_dict = {
            "metadata": [],
            "video": None,
            "audio_speak": None,
            "audio_hear": None,
            "mouse_movement": None,
            "key_press": None,
            "transcript_speak": [],
            "transcript_hear": [],
        }

        video_segments = []
        audio_in_segments = []
        audio_out_segments = []
        mouse_segments = []
        key_segments = []
        transcripts_in_list = []
        transcripts_out_list = []
        total_covered_frames = 0

        for c in covered_sessions:
            session_id = c["session_id"]
            s_start = c["start_frame_in_session"]
            s_end = c["end_frame_in_session"]
            sub_session_frames = s_end - s_start

            # Load session_info with neutral wording + accurate fallback log if needed
            try:
                session_info = self._load_session_info(session_id)
            except Exception as e:
                logging.error(
                    f"Node {self.node_id}, GPU {self.gpu_id}: Error loading session info for session {session_id}: "
                    f"{type(e).__name__}: {e}. Retrying with direct DB lookup."
                )
                session_info = self._load_session_info_from_db(session_id)

            frame_length_ms = 1000 // self.LATENT_FPS  # 100
            sess_start_timestamp = s_start * frame_length_ms
            sess_end_timestamp = s_end * frame_length_ms

            metadata_entry = {
                "player_id": session_info["player_id"],
                "player_name": session_info["player_name"],
                "player_email": session_info["player_email"],
                "player_gender": session_info.get("player_gender"),
                "player_skill_level": session_info.get("player_skill_level"),
                "session_id": session_id,
                "session_start_timestamp": session_info["start_time"],
                "start_frame": s_start,
                "end_frame": s_end,
                "window_length_frames": sub_session_frames
            }
            data_dict["metadata"].append(metadata_entry)

            # ---------- VIDEO ----------
            if "video" in self.modalities and session_info["modality_flags"]["video"]:
                primary_video_path = session_info["paths"]["video_encodings"]
                try:
                    vdata = self._load_video_encodings(primary_video_path, s_start, s_end, session_id)
                except Exception as e:
                    self._log_load_failure("video", primary_video_path, session_id, e, context="primary")
                    fb_info = self._load_session_info_from_db(session_id)
                    fb_video_path = fb_info["paths"]["video_encodings"]
                    try:
                        vdata = self._load_video_encodings(fb_video_path, s_start, s_end, session_id)
                    except Exception as e2:
                        self._log_load_failure("video", fb_video_path, session_id, e2, context="fallback")
                        raise
                video_segments.append(vdata)

            # ---------- AUDIO SPEAK + transcripts_speak ----------
            if "audio_speak" in self.modalities and session_info["modality_flags"]["audio_speak"]:
                primary_a_in_path = session_info["paths"]["audio_in_encodings"]
                try:
                    a_in_data = self._load_audio_encodings(primary_a_in_path, sess_start_timestamp, sess_end_timestamp)
                except Exception as e:
                    self._log_load_failure("audio_speak", primary_a_in_path, session_id, e, context="primary")
                    fb_info = self._load_session_info_from_db(session_id)
                    fb_a_in_path = fb_info["paths"]["audio_in_encodings"]
                    try:
                        a_in_data = self._load_audio_encodings(fb_a_in_path, sess_start_timestamp, sess_end_timestamp)
                    except Exception as e2:
                        self._log_load_failure("audio_speak", fb_a_in_path, session_id, e2, context="fallback")
                        raise
                audio_in_segments.append(a_in_data)

                try:
                    trans_in = self._load_transcripts(session_info["paths"]["db"], session_id, "transcript_in",
                                                      sess_start_timestamp, sess_end_timestamp)
                except Exception as e:
                    self._log_load_failure("transcript_speak(db)", session_info["paths"]["db"], session_id, e, context="primary")
                    fb_info = self._load_session_info_from_db(session_id)
                    fb_db = fb_info["paths"]["db"]
                    try:
                        trans_in = self._load_transcripts(fb_db, session_id, "transcript_in",
                                                          sess_start_timestamp, sess_end_timestamp)
                    except Exception as e2:
                        self._log_load_failure("transcript_speak(db)", fb_db, session_id, e2, context="fallback")
                        raise
                transcripts_in_list.extend(trans_in)

            # ---------- AUDIO HEAR + transcripts_hear ----------
            if "audio_hear" in self.modalities and session_info["modality_flags"]["audio_hear"]:
                primary_a_out_path = session_info["paths"]["audio_out_encodings"]
                try:
                    a_out_data = self._load_audio_encodings(primary_a_out_path, sess_start_timestamp, sess_end_timestamp)
                except Exception as e:
                    self._log_load_failure("audio_hear", primary_a_out_path, session_id, e, context="primary")
                    fb_info = self._load_session_info_from_db(session_id)
                    fb_a_out_path = fb_info["paths"]["audio_out_encodings"]
                    try:
                        a_out_data = self._load_audio_encodings(fb_a_out_path, sess_start_timestamp, sess_end_timestamp)
                    except Exception as e2:
                        self._log_load_failure("audio_hear", fb_a_out_path, session_id, e2, context="fallback")
                        raise
                audio_out_segments.append(a_out_data)

                try:
                    trans_out = self._load_transcripts(session_info["paths"]["db"], session_id, "transcript_out",
                                                       sess_start_timestamp, sess_end_timestamp)
                except Exception as e:
                    self._log_load_failure("transcript_hear(db)", session_info["paths"]["db"], session_id, e, context="primary")
                    fb_info = self._load_session_info_from_db(session_id)
                    fb_db = fb_info["paths"]["db"]
                    try:
                        trans_out = self._load_transcripts(fb_db, session_id, "transcript_out",
                                                           sess_start_timestamp, sess_end_timestamp)
                    except Exception as e2:
                        self._log_load_failure("transcript_hear(db)", fb_db, session_id, e2, context="fallback")
                        raise
                transcripts_out_list.extend(trans_out)

            # ---------- ACTION (mouse + key) ----------
            should_load_mouse = "mouse_movement" in self.modalities and session_info["modality_flags"].get("mouse")
            should_load_key = "key_press" in self.modalities and session_info["modality_flags"].get("keyboard")
            if should_load_mouse or should_load_key:
                primary_db = session_info["paths"]["db"]
                mouse_tensor = None
                key_tensor = None
                try:
                    con = sqlite3.connect(f'file:{primary_db}?mode=ro', uri=True)
                    if should_load_mouse:
                        loaded_mouse = self._load_mouse_movement(
                            con,
                            session_id,
                            sess_start_timestamp,
                            sess_end_timestamp,
                            sub_session_frames,
                        )
                        if torch.any(loaded_mouse != 0):
                            mouse_tensor = loaded_mouse
                    if should_load_key:
                        key_tensor = self._load_key_press_encoding(
                            con,
                            session_id,
                            sess_start_timestamp,
                            sess_end_timestamp,
                            sub_session_frames,
                        )
                    con.close()
                except Exception as e:
                    self._log_load_failure("action(db)", primary_db, session_id, e, context="primary")
                    fb_info = self._load_session_info_from_db(session_id)
                    fb_db = fb_info["paths"]["db"]
                    try:
                        con = sqlite3.connect(f'file:{fb_db}?mode=ro', uri=True)
                        if should_load_mouse:
                            loaded_mouse = self._load_mouse_movement(
                                con,
                                session_id,
                                sess_start_timestamp,
                                sess_end_timestamp,
                                sub_session_frames,
                            )
                            if torch.any(loaded_mouse != 0):
                                mouse_tensor = loaded_mouse
                        if should_load_key:
                            key_tensor = self._load_key_press_encoding(
                                con,
                                session_id,
                                sess_start_timestamp,
                                sess_end_timestamp,
                                sub_session_frames,
                            )
                        con.close()
                    except Exception as e2:
                        self._log_load_failure("action(db)", fb_db, session_id, e2, context="fallback")
                        raise

                if "mouse_movement" in self.modalities and should_load_mouse and mouse_tensor is not None:
                    mouse_segments.append(mouse_tensor)
                if "key_press" in self.modalities and should_load_key and key_tensor is not None:
                    key_segments.append(key_tensor)

            total_covered_frames += sub_session_frames

        # ----- Final assembly (unchanged behavior) -----
        if "video" in self.modalities:
            if video_segments:
                try:
                    data_dict["video"] = torch.cat(video_segments, dim=0)
                except Exception:
                    data_dict["video"] = torch.zeros((window_length_frames, 4, 96, 160), dtype=torch.float32)
            else:
                data_dict["video"] = torch.zeros((window_length_frames, 4, 96, 160), dtype=torch.float32)

        if "audio_speak" in self.modalities:
            final_expected = math.ceil((window_length_frames / self.LATENT_FPS) * self.AUDIO_TOKEN_FRAME_RATE)
            if audio_in_segments:
                try:
                    concat_in = torch.cat(audio_in_segments, dim=1)
                except Exception:
                    concat_in = None
                if concat_in is None:
                    data_dict["audio_speak"] = torch.zeros((128, final_expected), dtype=torch.float32)
                else:
                    current_tokens = concat_in.shape[1]
                    if current_tokens > final_expected:
                        concat_in = concat_in[:, :final_expected]
                    if current_tokens < final_expected:
                        pad = torch.zeros((128, final_expected - current_tokens), dtype=concat_in.dtype)
                        concat_in = torch.cat([concat_in, pad], dim=1)
                    data_dict["audio_speak"] = concat_in
            else:
                data_dict["audio_speak"] = torch.zeros((128, final_expected), dtype=torch.float32)

        if "audio_hear" in self.modalities:
            final_expected = math.ceil((window_length_frames / self.LATENT_FPS) * self.AUDIO_TOKEN_FRAME_RATE)
            if audio_out_segments:
                try:
                    concat_out = torch.cat(audio_out_segments, dim=1)
                except Exception:
                    concat_out = None
                if concat_out is None:
                    data_dict["audio_hear"] = torch.zeros((128, final_expected), dtype=torch.float32)
                else:
                    current_tokens = concat_out.shape[1]
                    if current_tokens > final_expected:
                        concat_out = concat_out[:, :final_expected]
                    if current_tokens < final_expected:
                        pad = torch.zeros((128, final_expected - current_tokens), dtype=concat_out.dtype)
                        concat_out = torch.cat([concat_out, pad], dim=1)
                    data_dict["audio_hear"] = concat_out
            else:
                data_dict["audio_hear"] = torch.zeros((128, final_expected), dtype=torch.float32)

        if "mouse_movement" in self.modalities:
            if mouse_segments:
                try:
                    mm = torch.cat(mouse_segments, dim=1)
                    data_dict["mouse_movement"] = mm
                except Exception:
                    data_dict["mouse_movement"] = torch.zeros(
                        (2, self.BINS_PER_FRAME * window_length_frames), dtype=torch.float32
                    )
            else:
                data_dict["mouse_movement"] = torch.zeros(
                    (2, self.BINS_PER_FRAME * window_length_frames), dtype=torch.float32
                )
        else:
            data_dict["mouse_movement"] = None

        if "key_press" in self.modalities:
            if key_segments:
                try:
                    kp = torch.cat(key_segments, dim=1)
                    data_dict["key_press"] = kp
                except Exception:
                    data_dict["key_press"] = torch.zeros(
                        (16, self.KEY_PRESS_ENC_PER_FRAME * window_length_frames), dtype=torch.float32
                    )
            else:
                data_dict["key_press"] = torch.zeros(
                    (16, self.KEY_PRESS_ENC_PER_FRAME * window_length_frames), dtype=torch.float32
                )
        else:
            data_dict["key_press"] = None

        data_dict["transcript_speak"] = transcripts_in_list
        data_dict["transcript_hear"] = transcripts_out_list
            
        return FullData(batch=data_dict)

    def _load_transcripts(self, db_path, session_id, table_name, start_timestamp, end_timestamp):
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

    def _load_video_encodings(self, encodings_path, start_frame, end_frame, session_id):
        # encodings_path is now the full path to the .hdf5 file
        if not encodings_path.exists():
            raise FileNotFoundError(f"Video HDF5 file not found for session {session_id} at {encodings_path}")

        try:
            with h5py.File(encodings_path, "r") as f:
                # 1. Read the specific slice of frames
                # HDF5 allows direct slicing like numpy arrays
                latents_np = f['latents'][start_frame:end_frame]
                min_vals_np = f['min_vals'][start_frame:end_frame]
                scales_np = f['scales'][start_frame:end_frame]
            
            # 2. Convert to Torch Tensors
            sliced_latents = torch.from_numpy(latents_np)
            sliced_min_vals = torch.from_numpy(min_vals_np)
            sliced_scales = torch.from_numpy(scales_np)

            # 3. Dequantize
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
        audio_encodings_path = Path(audio_encodings_path)
        if not audio_encodings_path.exists():
            raise FileNotFoundError(f"Audio HDF5 not found: {audio_encodings_path}")
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
            # if current_tokens < expected_audio_tokens:
            #     pad = torch.zeros((128, expected_audio_tokens - current_tokens), dtype=tokens.dtype)
            #     tokens = torch.cat([tokens, pad], dim=-1)
            return tokens

    def _load_mouse_movement(self, con, session_id, start_timestamp, end_timestamp, sub_session_frames):
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
        binned_dx = np.where((binned_dx < self.CLIP_MOUSE_DX[0]) | (binned_dx > self.CLIP_MOUSE_DX[1]), 0.0, binned_dx)
        binned_dy = np.where((binned_dy < self.CLIP_MOUSE_DY[0]) | (binned_dy > self.CLIP_MOUSE_DY[1]), 0.0, binned_dy)
        binned_data = np.concatenate([binned_dx.reshape(1, -1), binned_dy.reshape(1, -1)], axis=0)
        return torch.tensor(binned_data, dtype=torch.float32)

    def _load_key_press_encoding(self, con, session_id, start_timestamp, end_timestamp, sub_session_frames):
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

    def collate_fn(self, batch):
        collated = {
            "metadata": [],
            "video": [],
            "audio_speak": [],
            "audio_hear": [],
            "mouse_movement": [],
            "key_press": [],
            "transcript_speak": [],
            "transcript_hear": [],
            "dataframe_indices": [],
        }
        for item in batch:
            # Convert FullData to dict for processing
            item_dict = item.to_dict() if isinstance(item, FullData) else item
            collated["metadata"].append(item_dict["metadata"])
            if "video" in self.modalities:
                # Reshape video from (T*2, C, H, W) to (T, 2, C, H, W)
                t = item_dict["video"].shape[0] // 2
                collated["video"].append(rearrange(item_dict["video"], "(t 2) c h w -> t 2 c h w", t=t))
            if "audio_speak" in self.modalities:
                # Reshape audio from (C, T*15) to (T, 15, C)
                t = item_dict["audio_speak"].shape[1] // 15
                collated["audio_speak"].append(rearrange(item_dict["audio_speak"], "c (t 15) -> t 15 c", t=t))
            if "audio_hear" in self.modalities:
                t = item_dict["audio_hear"].shape[1] // 15
                collated["audio_hear"].append(rearrange(item_dict["audio_hear"], "c (t 15) -> t 15 c", t=t))
            if "mouse_movement" in self.modalities and item_dict["mouse_movement"] is not None:
                # Reshape mouse from (2, T*10) to (T, 20, 2)
                mm = item_dict["mouse_movement"].permute(1, 0).reshape(
                    -1, MOUSE_TOKENS_PER_UNIT, item_dict["mouse_movement"].shape[0]
                )
                collated["mouse_movement"].append(mm)

            if "key_press" in self.modalities and item_dict["key_press"] is not None:
                # Reshape keypress from (16, T*5) to (T, 10, 16)
                kp = item_dict["key_press"].permute(1, 0).reshape(
                    -1, KEYBOARD_TOKENS_PER_UNIT, item_dict["key_press"].shape[0]
                )
                collated["key_press"].append(kp)
            collated["transcript_speak"].append(item_dict["transcript_speak"])
            collated["transcript_hear"].append(item_dict["transcript_hear"])

            item_metadata = item_dict["metadata"]
            item_df_segments = []
            for segment in item_metadata:
                segment_start_frame = int(segment["start_frame"])
                segment_end_frame = int(segment["end_frame"])
                seg_start_df = segment_start_frame // 2
                seg_end_df = segment_end_frame // 2
                item_df_segments.append(
                    torch.arange(
                        seg_start_df,
                        seg_end_df,
                        dtype=torch.long,
                    )
                )

            if item_df_segments:
                item_dataframe_indices = torch.cat(item_df_segments, dim=0)
            else:
                item_dataframe_indices = torch.empty((0,), dtype=torch.long)

            collated["dataframe_indices"].append(item_dataframe_indices)
        if "video" in self.modalities:
            collated["video"] = pad_sequence_dims(collated["video"], (0,), padding_value=0.0)
        if "audio_speak" in self.modalities:
            collated["audio_speak"] = pad_sequence_dims(collated["audio_speak"], (0,), padding_value=0.0)
        if "audio_hear" in self.modalities:
            collated["audio_hear"] = pad_sequence_dims(collated["audio_hear"], (0,), padding_value=0.0)
        if "mouse_movement" in self.modalities and collated["mouse_movement"]:
            collated["mouse_movement"] = pad_sequence_dims(collated["mouse_movement"], (0,), padding_value=0.0)
        else:
            collated["mouse_movement"] = None

        if "key_press" in self.modalities and collated["key_press"]:
            collated["key_press"] = pad_sequence_dims(collated["key_press"], (0,), padding_value=0.0)
        else:
            collated["key_press"] = None
        collated["dataframe_indices"] = pad_sequence_dims(collated["dataframe_indices"], (0,), padding_value=-1)
        
        if self.use_ssd_cache:
            with self.counter_lock:
                self.batch_counter.value += 1
        return FullData(batch=collated)

    def __del__(self):
        if hasattr(self, "connection") and self.connection:
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None
        if self.use_ssd_cache:
            self.cleanup_cache()
            if hasattr(self, "pending_ready_event"):
                self.pending_ready_event.set()

    def cleanup_cache(self):
        with self.cache_lock:
            for cache in self.active_cache_pool:
                path = cache["ssd_cache_path"]
                try:
                    shutil.rmtree(path)
                    logging.info(f"Node {self.node_id}, GPU {self.gpu_id}: Cleanup: Removed cache at {path}")
                except Exception as e:
                    logging.error(f"Node {self.node_id}, GPU {self.gpu_id}: Cleanup: Error removing cache at {path}: {e}")
            self.active_cache_pool.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["connection"] = None
        return state
    
    def _reseed(self, worker_seed: int):
        self.rng = random.Random(worker_seed + getattr(self, "rank", 0))
    
    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = worker_info.seed  # unique per worker and epoch
        ds = worker_info.dataset
        ds._reseed(worker_seed)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))

    @staticmethod
    def add_command_line_options(argparser):
        argparser.add_argument('--dataset_path', type=str, default=None,
                               help='Path to the dataset folder containing session data.')
        argparser.add_argument('--modalities', type=str, nargs='+', default=["video", "audio_speak", "audio_hear", "key_press", "mouse_movement"],
                               help='List of modalities to load.')
        argparser.add_argument('--player_names', type=str, nargs='*', default=None,
                               help='List of player names whose data to load.')
        argparser.add_argument('--global_database_path', type=str,
                               default=None,
                               help='Path to the global database file.')
        argparser.add_argument('--use_ssd_cache', action='store_true',
                               help='If set, use SSD caching by copying a chunk to the specified SSD cache directory.')
        argparser.add_argument('--ssd_cache_dir', type=str, default=os.environ.get("SLURM_TMPDIR", "/tmp"),
                               help='Base directory for SSD cache. Defaults to $SLURM_TMPDIR or "/tmp".')
        argparser.add_argument('--chunk_size_gb', type=float, default=1.0,
                               help='Size (in GB) of each chunk for SSD caching. Default is 1GB.')
        argparser.add_argument('--cache_queue_size', type=int, default=2,
                               help='Number of SSD cache buffers to maintain for multi-buffering.')
        argparser.add_argument('--ssd_cache_removal_grace_period', type=int, default=60,
                               help='Grace period (in seconds) before removal of old cache buffers.')
        argparser.add_argument('--ssd_cache_refresh_interval', type=int, default=5,
                               help='Interval (in seconds) between asynchronous cache refresh iterations.')
        argparser.add_argument(
            '--iterations_per_cache', type=int, default=1000,
            help='Exactly how many Dataset iterations to perform with the current '
                'SSD-cache pool before the pool is swapped out.'
        )

if __name__ == "__main__":
    import argparse
    import traceback
    from data.samplers.dynamic_length import DynamicLengthBatchSampler

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Plaicraft Iter Dataset Loader (with SSD cache + dynamic length sampler)")

    IterStyleDataset.add_command_line_options(parser)
    DynamicLengthBatchSampler.add_command_line_options(parser)

    # DataLoader / testing options
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Nominal batch size used only for logging / intuition; DynamicLengthBatchSampler controls actual batch shapes.'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of worker processes for data loading.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Base random seed for dataset and sampler.'
    )
    parser.add_argument(
        '--num_batches_to_test',
        type=int,
        default=100,
        help='Number of batches to iterate over before stopping.'
    )

    args = parser.parse_args()

    try:
        dataset = IterStyleDataset(args=args, seed=args.seed)
        batch_sampler = DynamicLengthBatchSampler(args=args, seed=args.seed)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
            worker_init_fn=IterStyleDataset.worker_init_fn,
            persistent_workers=True
        )

        for batch_idx, batch_data in enumerate(data_loader):
            print(f"\n=== Batch {batch_idx} ===")
            batch_dict = batch_data.to_dict()
            print("Metadata in this batch:")
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

            if batch_idx + 1 == args.num_batches_to_test:
                break

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()
