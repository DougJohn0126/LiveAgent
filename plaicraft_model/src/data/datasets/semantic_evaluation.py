#!/usr/bin/env python3
# data/components/semantic_evaluation_dataset.py

import math
import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from data.data_classes import FullData
from utils.constants import (
    AUDIO_TOKEN_FPS,
    KEYBOARD_TOKENS_PER_UNIT,
    KEYBOARD_TOKENS_PER_VIDEO_FRAME,
    MOUSE_TOKENS_PER_UNIT,
    MOUSE_TOKENS_PER_VIDEO_FRAME,
    VIDEO_FPS,
    VIDEO_FRAMES_PER_UNIT,
)


class SemanticEvaluationDataset(Dataset):
    """
    Semantic evaluation dataset that, for each CSV row, returns TWO aligned windows:

            • context window: [0, R_start]
            • target  window: [R_start, R_start + R_dur]

    Both windows are snapped to the latent grid.
    """

    LATENT_FPS = VIDEO_FPS
    VIDEO_LATENTS_PER_BATCH = 100
    AUDIO_TOKEN_FRAME_RATE = AUDIO_TOKEN_FPS
    USE_FP16 = True

    BINS_PER_FRAME = MOUSE_TOKENS_PER_VIDEO_FRAME
    CLIP_MOUSE_DX = (-150.0, 150.0)
    CLIP_MOUSE_DY = (-100.0, 100.0)
    KEY_PRESS_ENC_PER_FRAME = KEYBOARD_TOKENS_PER_VIDEO_FRAME
    MODALITIES = ["video", "audio_speak", "audio_hear", "key_press", "mouse_movement"]

    def __init__(
        self,
        dataset_path: str,
        semantic_evaluation_db_path: str,
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.semantic_evaluation_db_path = Path(semantic_evaluation_db_path)
        self.modalities = list(self.MODALITIES)
        assert self.dataset_path.is_dir(), f"dataset_path not found: {self.dataset_path}"
        assert self.semantic_evaluation_db_path.is_file(), f"semantic_evaluation_db_path not found: {self.semantic_evaluation_db_path}"

        self.items = self._read_db_and_index(self.semantic_evaluation_db_path)

    @staticmethod
    def _truthy(x: str) -> bool:
        if x is None:
            return False
        s = str(x).strip().lower()
        return s in ("true", "1", "yes", "y", "t")

    def _read_db_and_index(self, db_path: Path) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = con.cursor()
            cur.execute(
                """
                SELECT
                    "Num",
                    "Session_ID",
                    "R_start (ms)",
                    "R_duration (ms)",
                    "Test Type",
                    "Metric",
                    "P_R_valid",
                    player_email,
                    player_id,
                    player_name,
                    player_gender,
                    player_skill_level
                FROM prompts
                ORDER BY CAST("Num" AS INTEGER), rowid
                """
            )
            rows = cur.fetchall()
        finally:
            con.close()

        for row in rows:
            (
                num_raw,
                sid_raw,
                r_start_raw,
                r_duration_raw,
                test_type,
                metric,
                p_r_valid_raw,
                p_email_raw,
                player_id_raw,
                player_name_raw,
                player_gender_raw,
                player_skill_level_raw,
            ) = row

            if not self._truthy(p_r_valid_raw):
                continue

            sid = str(sid_raw).strip() if sid_raw is not None else ""
            if not sid:
                continue

            p_email = str(p_email_raw).strip() if p_email_raw is not None else ""
            if not p_email:
                logging.warning("Session_ID=%s has empty player_email in prompts; skipping.", sid)
                continue

            if player_id_raw is None or str(player_id_raw).strip() == "":
                logging.warning("Session_ID=%s missing player_id in prompts; skipping.", sid)
                continue

            try:
                player_id = int(str(player_id_raw).strip())
            except Exception:
                logging.warning("Session_ID=%s has invalid player_id=%s; skipping.", sid, player_id_raw)
                continue

            player_name = "" if player_name_raw is None else str(player_name_raw).strip()
            player_gender = "" if player_gender_raw is None else str(player_gender_raw).strip()
            player_skill_level = "" if player_skill_level_raw is None else str(player_skill_level_raw).strip()

            if not player_name:
                logging.warning("Session_ID=%s missing player_name in prompts; skipping.", sid)
                continue

            try:
                r_start_ms = int(float(r_start_raw))
            except Exception:
                logging.warning("Bad R_start (ms) for Session_ID=%s; skipping.", sid)
                continue

            try:
                num = int(float(num_raw))
            except Exception:
                logging.warning("Bad Num for Session_ID=%s; skipping.", sid)
                continue

            try:
                r_duration_ms = int(float(r_duration_raw if r_duration_raw is not None else 0))
            except Exception:
                r_duration_ms = 0

            sess_dir = self.dataset_path / p_email / sid
            if not sess_dir.is_dir():
                logging.warning("Session folder not found: %s (skipping)", sess_dir)
                continue

            items.append(dict(
                player_email=p_email,
                session_id=sid,
                num=num,
                r_start_ms=r_start_ms,
                r_duration_ms=r_duration_ms,
                test_type=test_type or "",
                metric=metric or "",
                session_start_timestamp=0,
                player_name=player_name,
                player_id=player_id,
                player_gender=player_gender,
                player_skill_level=player_skill_level,
            ))

        if not items:
            raise ValueError("No valid rows after filtering prompts and session folder checks.")
        return items

    # The rest of the implementation mirrors the original semantic evaluation dataset with
    # the same method names but using semantic_evaluation_db_path where appropriate.

    @staticmethod
    def _align_up(ms: int, step: int) -> int:
        return int(math.ceil(max(0, ms) / step) * step)

    @staticmethod
    def _align_down(ms: int, step: int) -> int:
        return int((max(0, ms) // step) * step)

    def _prompt_bounds(self, r_start_ms: int) -> Tuple[int, int, int, int]:
        start_frame = 0
        frames = int(math.floor((r_start_ms / 1000.0) * self.LATENT_FPS))
        frames = max(VIDEO_FRAMES_PER_UNIT, (frames // VIDEO_FRAMES_PER_UNIT) * VIDEO_FRAMES_PER_UNIT)

        end_frame = frames
        start_ms = 0
        end_ms = end_frame * 100
        return start_frame, end_frame, start_ms, end_ms

    def _response_bounds(self, r_start_ms: int, r_duration_ms: int, prompt_end_ms: int) -> Tuple[int, int, int, int]:
        start_ms = prompt_end_ms
        raw_end_ms = max(r_start_ms, 0) + max(r_duration_ms, 0)
        end_ms = self._align_down(raw_end_ms, 200)
        if end_ms <= start_ms:
            end_ms = start_ms + 200

        if ((end_ms - start_ms) % 200) != 0:
            end_ms = self._align_down(end_ms, 200)

        start_frame = start_ms // 100
        end_frame = end_ms // 100

        return start_frame, end_frame, start_ms, end_ms

    def _build_window_payload(
        self,
        *,
        p_email: str,
        player_id: int,
        player_name: str,
        player_gender: str,
        player_skill_level: str,
        session_start_timestamp: int,
        sid: str,
        num: int,
        window_type: str,
        start_frame: int,
        end_frame: int,
        start_ms: int,
        end_ms: int,
        test_type: str,
        metric: str,
        r_start_ms: int,
        r_duration_ms: int,
    ) -> Dict[str, Any]:
        sess_dir = self.dataset_path / p_email / sid

        paths = dict(
            video_encodings=sess_dir / "encoded_video_hdf5" / f"{sid}_encoded_video.hdf5",
            audio_in_encodings=sess_dir / "encoded_audio_continuous" / f"{sid}_encoded_audio_in.hdf5",
            audio_out_encodings=sess_dir / "encoded_audio_continuous" / f"{sid}_encoded_audio_out.hdf5",
            db=sess_dir / f"{sid}.db",
        )
        modality_flags = dict(
            video=paths["video_encodings"].is_file(),
            audio_speak=paths["audio_in_encodings"].is_file(),
            audio_hear=paths["audio_out_encodings"].is_file(),
            mouse=True,
            keyboard=True,
        )

        data = {
            "metadata": [],
            "video": [],
            "audio_speak": [],
            "audio_hear": [],
            "mouse_movement": [],
            "key_press": [],
            "transcript_speak": [],
            "transcript_hear": [],
        }

        data["metadata"].append({
            "player_id": int(player_id),
            "player_name": player_name,
            "player_email": p_email,
            "player_gender": player_gender,
            "player_skill_level": player_skill_level,
            "num": num,
            "session_id": sid,
            "session_start_timestamp": int(session_start_timestamp),
            "window_type": window_type,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "window_length_frames": end_frame - start_frame,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "response_start_ms": r_start_ms,
            "response_duration_ms": r_duration_ms,
            "test_type": test_type,
            "metric": metric,
        })

        # ---- video ----
        if "video" in self.modalities:
            if modality_flags["video"]:
                try:
                    vid = self._load_video_encodings(paths["video_encodings"], start_frame, end_frame, sid)
                except Exception as e:
                    logging.error("Video load failed for %s: %s", sid, e)
                    vid = torch.zeros(
                        (end_frame - start_frame, 4, 96, 160),
                        dtype=torch.float16 if self.USE_FP16 else torch.float32
                    )
            else:
                vid = torch.zeros(
                    (end_frame - start_frame, 4, 96, 160),
                    dtype=torch.float16 if self.USE_FP16 else torch.float32
                )
            data["video"] = vid

        # ---- audio_speak / audio_hear ----
        if "audio_speak" in self.modalities:
            a_in = self._load_audio_encodings(paths["audio_in_encodings"], start_ms, end_ms)
            data["audio_speak"] = a_in
            data["transcript_speak"].extend(
                self._load_transcripts(paths["db"], sid, "transcript_in", start_ms, end_ms)
            )
        if "audio_hear" in self.modalities:
            a_out = self._load_audio_encodings(paths["audio_out_encodings"], start_ms, end_ms)
            data["audio_hear"] = a_out
            data["transcript_hear"].extend(
                self._load_transcripts(paths["db"], sid, "transcript_out", start_ms, end_ms)
            )

        # ---- actions ----
        if "mouse_movement" in self.modalities or "key_press" in self.modalities:
            sub_len = end_frame - start_frame
            mouse_tensor = None
            key_tensor = None

            if paths["db"].is_file():
                try:
                    con = sqlite3.connect(f'file:{paths["db"]}?mode=ro', uri=True)
                    if "mouse_movement" in self.modalities:
                        loaded_mouse = self._load_mouse_movement(con, sid, start_ms, end_ms, sub_len)
                        if torch.any(loaded_mouse != 0):
                            mouse_tensor = loaded_mouse
                    if "key_press" in self.modalities:
                        key_tensor = self._load_key_press_encoding(con, sid, start_ms, end_ms, sub_len)
                    con.close()
                except Exception as e:
                    logging.error("Error loading action data for %s: %s", sid, e)

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

        for subk, bins, dim in (("mouse_movement", self.BINS_PER_FRAME, 2),
                                ("key_press", self.KEY_PRESS_ENC_PER_FRAME, 16)):
            if subk in self.modalities:
                if data[subk]:
                    data[subk] = torch.cat(data[subk], dim=1)
                else:
                    data[subk] = torch.zeros((dim, bins * (end_frame - start_frame)), dtype=torch.float32)
            else:
                data[subk] = None

        return data

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Any]]:
        it = self.items[idx]
        p_email = it["player_email"]
        sid = it["session_id"]
        player_id = int(it["player_id"])
        player_name = it["player_name"]
        player_gender = it["player_gender"]
        player_skill_level = it["player_skill_level"]
        session_start_timestamp = int(it["session_start_timestamp"])
        num = it["num"]
        r_start_ms = int(it["r_start_ms"])
        r_duration_ms = int(it.get("r_duration_ms", 0))
        test_type = it.get("test_type", "")
        metric = it.get("metric", "")

        p_sframe, p_eframe, p_sms, p_ems = self._prompt_bounds(r_start_ms)
        r_sframe, r_eframe, r_sms, r_ems = self._response_bounds(r_start_ms, r_duration_ms, p_ems)

        context = self._build_window_payload(
            p_email=p_email,
            player_id=player_id,
            player_name=player_name,
            player_gender=player_gender,
            player_skill_level=player_skill_level,
            session_start_timestamp=session_start_timestamp,
            sid=sid,
            num=num,
            window_type="context",
            start_frame=p_sframe, end_frame=p_eframe,
            start_ms=p_sms, end_ms=p_ems,
            test_type=test_type, metric=metric,
            r_start_ms=r_start_ms, r_duration_ms=r_duration_ms,
        )
        target = self._build_window_payload(
            p_email=p_email,
            player_id=player_id,
            player_name=player_name,
            player_gender=player_gender,
            player_skill_level=player_skill_level,
            session_start_timestamp=session_start_timestamp,
            sid=sid,
            num=num,
            window_type="target",
            start_frame=r_sframe, end_frame=r_eframe,
            start_ms=r_sms, end_ms=r_ems,
            test_type=test_type, metric=metric,
            r_start_ms=r_start_ms, r_duration_ms=r_duration_ms,
        )

        return {"context": FullData(batch=context), "target": FullData(batch=target)}

    def _collate_one_window(self, batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated = {
            "metadata": [],
            "video": [],
            "audio_speak": [],
            "audio_hear": [],
            "mouse_movement": [],
            "key_press": [],
            "transcript_speak": [],
            "transcript_hear": [],
        }

        for item in batch_list:
            item_dict = item.to_dict() if isinstance(item, FullData) else item
            collated["metadata"].append(item_dict["metadata"])

            if "video" in self.modalities and item_dict["video"] is not None:
                video = item_dict["video"]
                collated["video"].append(video.reshape(-1, 2, *video.shape[1:]))

            if "audio_speak" in self.modalities and item_dict["audio_speak"] is not None:
                audio_speak = item_dict["audio_speak"]
                collated["audio_speak"].append(
                    audio_speak.permute(1, 0).reshape(-1, 15, audio_speak.shape[0])
                )

            if "audio_hear" in self.modalities and item_dict["audio_hear"] is not None:
                audio_hear = item_dict["audio_hear"]
                collated["audio_hear"].append(
                    audio_hear.permute(1, 0).reshape(-1, 15, audio_hear.shape[0])
                )

            if "mouse_movement" in self.modalities:
                mm = item_dict["mouse_movement"]
                if mm is not None:
                    mm = mm.permute(1, 0).reshape(-1, MOUSE_TOKENS_PER_UNIT, mm.shape[0])
                    collated["mouse_movement"].append(mm)
                else:
                    collated["mouse_movement"].append(None)

            if "key_press" in self.modalities:
                kp = item_dict["key_press"]
                if kp is not None:
                    kp = kp.permute(1, 0).reshape(-1, KEYBOARD_TOKENS_PER_UNIT, kp.shape[0])
                    collated["key_press"].append(kp)
                else:
                    collated["key_press"].append(None)

            collated["transcript_speak"].append(item_dict["transcript_speak"])
            collated["transcript_hear"].append(item_dict["transcript_hear"])

        def _safe_stack(lst, dim=0):
            lst = [x for x in lst if x is not None]
            if not lst:
                return None
            return torch.stack(lst, dim=dim)

        for mod in ["video", "audio_speak", "audio_hear"]:
            try:
                collated[mod] = _safe_stack(collated[mod], dim=0)
            except Exception as e:
                logging.error(f"Error stacking modality '{mod}': {e}")
                collated[mod] = None

        mm_list = [x for x in collated["mouse_movement"] if x is not None]
        kp_list = [x for x in collated["key_press"] if x is not None]
        collated["mouse_movement"] = torch.stack(mm_list, dim=0) if mm_list else None
        collated["key_press"] = torch.stack(kp_list, dim=0) if kp_list else None
        return FullData(batch=collated)

    def collate_fn(self, batch: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        context_list = [x["context"] for x in batch]
        target_list = [x["target"] for x in batch]

        context = self._collate_one_window(context_list)
        target = self._collate_one_window(target_list)

        return {
            "context": context,
            "target": target,
        }

    def _load_transcripts(self, db_path: Path, session_id: str, table_name: str, start_ms: int, end_ms: int):
        if not db_path.is_file():
            return []
        try:
            con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
            cur = con.cursor()
            cur.execute(f"""
                SELECT word, start_time, end_time
                FROM {table_name}
                WHERE session_id=?
                  AND (start_time >= ? AND end_time <= ?)
            """, (session_id, start_ms, end_ms))
            rows = cur.fetchall()
            con.close()
            return rows
        except Exception:
            return []

    def _load_video_encodings(self, enc_path: Path, start_frame: int, end_frame: int, session_id: str):
        if not enc_path.exists() or enc_path.suffix.lower() != ".hdf5":
            raise FileNotFoundError(
                f"Video HDF5 file not found for session {session_id} at {enc_path}"
            )

        try:
            with h5py.File(enc_path, "r") as f:
                latents_np = f["latents"][start_frame:end_frame]
                min_vals_np = f["min_vals"][start_frame:end_frame]
                scales_np = f["scales"][start_frame:end_frame]
            sliced_latents = torch.from_numpy(latents_np)
            sliced_min_vals = torch.from_numpy(min_vals_np)
            sliced_scales = torch.from_numpy(scales_np)
            dequantized_latents = self._dequantize_from_int8(sliced_latents, sliced_min_vals, sliced_scales)
            return dequantized_latents
        except Exception as e:
            raise RuntimeError(f"Failed to load video HDF5 for session {session_id}: {e}")

    def _dequantize_from_int8(self, q: torch.Tensor, mn: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
        while mn.dim() < q.dim():
            mn = mn.unsqueeze(-1)
        while sc.dim() < q.dim():
            sc = sc.unsqueeze(-1)
        out = q.to(torch.float32) * sc + mn
        return out.half() if self.USE_FP16 else out

    def _load_audio_encodings(self, path: Path, start_ms: int, end_ms: int) -> torch.Tensor:
        expected_tokens = max(1, math.ceil((max(end_ms - start_ms, 0) / 1000.0) * self.AUDIO_TOKEN_FRAME_RATE))

        if not path.is_file():
            return torch.zeros(
                (128, expected_tokens),
                dtype=torch.float16 if self.USE_FP16 else torch.float32
            )

        try:
            with h5py.File(path, "r") as hf:
                ds = hf["audio_latents"]
                start_tok = int(start_ms / 1000.0 * self.AUDIO_TOKEN_FRAME_RATE)
                end_tok = start_tok + expected_tokens
                arr = ds[0, :, start_tok:end_tok]
                tokens = torch.tensor(arr)
            if tokens.dim() == 3 and tokens.size(0) == 1:
                tokens = tokens.squeeze(0)
            elif tokens.dim() != 2:
                raise ValueError(f"Unexpected audio tokens shape: {tokens.shape}")

            cur = tokens.shape[-1]
            if cur > expected_tokens:
                tokens = tokens[:, :expected_tokens]
            elif cur < expected_tokens:
                pad = torch.zeros((128, expected_tokens - cur), dtype=tokens.dtype)
                tokens = torch.cat([tokens, pad], dim=-1)

            if self.USE_FP16:
                tokens = tokens.half()
            else:
                tokens = tokens.float()
            return tokens
        except Exception as e:
            logging.error("Error loading audio encodings from %s: %s", path, e)
            return torch.zeros(
                (128, expected_tokens),
                dtype=torch.float16 if self.USE_FP16 else torch.float32
            )

    def _load_mouse_movement(self, con, session_id, start_ms, end_ms, sub_frames):
        try:
            cur = con.cursor()
            cur.execute("""
                SELECT timestamp, mouseDX, mouseDY
                FROM mouse_movement
                WHERE session_id=? AND timestamp >= ? AND timestamp < ?
                ORDER BY timestamp ASC
            """, (session_id, start_ms, end_ms))
            rows = cur.fetchall()
            if not rows:
                return torch.zeros((2, self.BINS_PER_FRAME * sub_frames), dtype=torch.float32)

            events = np.array(rows, dtype=np.float32)

            frame_len = 1000.0 / self.LATENT_FPS
            dt = events[:, 0] - start_ms

            fidx = np.floor(dt / frame_len).astype(np.int32)
            fidx = np.clip(fidx, 0, sub_frames - 1)

            dt_in = dt - fidx * frame_len
            bin_w = frame_len / self.BINS_PER_FRAME
            bidx = np.floor(dt_in / bin_w).astype(np.int32)
            bidx = np.clip(bidx, 0, self.BINS_PER_FRAME - 1)

            ob = fidx * self.BINS_PER_FRAME + bidx
            total_bins = sub_frames * self.BINS_PER_FRAME

            bdx = np.bincount(ob, weights=events[:, 1], minlength=total_bins)
            bdy = np.bincount(ob, weights=events[:, 2], minlength=total_bins)

            bdx = bdx.reshape((sub_frames, self.BINS_PER_FRAME))
            bdy = bdy.reshape((sub_frames, self.BINS_PER_FRAME))

            bdx = np.where((bdx < self.CLIP_MOUSE_DX[0]) | (bdx > self.CLIP_MOUSE_DX[1]), 0.0, bdx)
            bdy = np.where((bdy < self.CLIP_MOUSE_DY[0]) | (bdy > self.CLIP_MOUSE_DY[1]), 0.0, bdy)

            out = np.concatenate([bdx.reshape(1, -1), bdy.reshape(1, -1)], axis=0)
            return torch.tensor(out, dtype=torch.float32)
        except Exception as e:
            logging.error("Error loading mouse movement: %s", e)
            return torch.zeros((2, self.BINS_PER_FRAME * sub_frames), dtype=torch.float32)

    def _load_key_press_encoding(self, con, session_id, start_ms, end_ms, sub_frames):
        try:
            cur = con.cursor()
            frame_len = 1000.0 / self.LATENT_FPS
            cur.execute("""
                SELECT start_timestamp, end_timestamp, encoding
                FROM key_press_encodings
                WHERE session_id=? AND end_timestamp >= ? AND start_timestamp < ?
                ORDER BY start_timestamp
            """, (session_id, start_ms, end_ms))
            rows = cur.fetchall()

            enc_map = {(st, et): enc_blob for (st, et, enc_blob) in rows}

            all_encs = []
            for frame_idx in range(sub_frames):
                f_start = int(start_ms + frame_len * frame_idx)
                f_end = int(f_start + frame_len)
                enc_blob = enc_map.get((f_start, f_end), None)
                if enc_blob is not None:
                    arr = pickle.loads(enc_blob)
                    enc_t = torch.tensor(arr, dtype=torch.float32)
                else:
                    enc_t = torch.zeros((16, self.KEY_PRESS_ENC_PER_FRAME), dtype=torch.float32)
                all_encs.append(enc_t)
            if not all_encs:
                return None
            return torch.cat(all_encs, dim=1)
        except Exception as e:
            logging.error("Error loading key_press encoding: %s", e)
            return None

    @staticmethod
    def add_command_line_options(p):
        p.add_argument("--dataset_path", type=str, default=None)
        p.add_argument("--semantic_evaluation_db_path", type=str, required=True)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser("Plaicraft Semantic Evaluation Dataset loader")
    SemanticEvaluationDataset.add_command_line_options(ap)
    args = ap.parse_args()

    ds = SemanticEvaluationDataset(
        dataset_path=args.dataset_path,
        semantic_evaluation_db_path=args.semantic_evaluation_db_path,
    )
    dl = DataLoader(
        ds,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        collate_fn=ds.collate_fn,
        persistent_workers=1,
    )

    for i, batch in enumerate(dl):
        print(f"\n=== Batch {i} ===")
        for which in ("context", "target"):
            window = batch[which]
            window_dict = window.to_dict()
            metadata = window_dict.get("metadata", None)
            if metadata and len(metadata) > 0 and len(metadata[0]) > 0:
                print(f"[{which.upper()}] Metadata:", metadata[0][0])
            else:
                print(f"[{which.upper()}] Metadata:", metadata)

            if window.video is not None:
                print(f"[{which}] Video:", window.video.shape)
            if window.audio_speak is not None:
                print(f"[{which}] Audio_speak:", window.audio_speak.shape)
            if window.audio_hear is not None:
                print(f"[{which}] Audio_hear:", window.audio_hear.shape)

            if window.mouse_movement is not None:
                print(f"[{which}] Mouse:", window.mouse_movement.shape)
            if window.key_press is not None:
                print(f"[{which}] Key:", window.key_press.shape)

        if i >= 50:
            break
