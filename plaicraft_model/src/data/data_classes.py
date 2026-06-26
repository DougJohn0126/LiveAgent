import torch
import numpy as np
from tensordict import tensorclass, TensorDict, NonTensorData
from typing import Any, Dict


@tensorclass
class FullData:
    video: torch.Tensor
    audio_speak: torch.Tensor
    audio_hear: torch.Tensor
    key_press: torch.Tensor
    mouse_movement: torch.Tensor
    metadata: Any
    dataframe_indices: torch.Tensor
    transcript_speak: Any
    transcript_hear: Any
    original_dtypes: Dict[str, Any]

    @staticmethod
    def _unwrap_non_tensor(value: Any) -> Any:
        if value is None or torch.is_tensor(value):
            return value
        if type(value).__name__ != "NonTensorData":
            return value

        for attr in ("data", "value", "obj", "_data"):
            if hasattr(value, attr):
                inner = getattr(value, attr)
                if callable(inner):
                    try:
                        inner = inner()
                    except Exception:
                        continue
                return inner

        try:
            return value.item()
        except Exception:
            return value

    @classmethod
    def _wrap_non_tensor(cls, value: Any) -> Any:
        if value is None or type(value).__name__ == "NonTensorData":
            return value
        return NonTensorData(value)

    def __init__(self, batch):
        self.video = batch.get("video", None)
        self.audio_speak = batch.get("audio_speak", None)
        self.audio_hear = batch.get("audio_hear", None)

        self.key_press = batch.get("key_press", None)
        self.mouse_movement = batch.get("mouse_movement", None)

        self.dataframe_indices = batch.get("dataframe_indices", None)
        if self.dataframe_indices is not None:
            if not torch.is_tensor(self.dataframe_indices):
                raise TypeError("dataframe_indices must be a torch.Tensor when provided.")
            if self.dataframe_indices.dtype != torch.long:
                raise TypeError(
                    f"dataframe_indices must be torch.long, got {self.dataframe_indices.dtype}."
                )

        self.metadata = self._wrap_non_tensor(self._unwrap_non_tensor(batch.get("metadata", None)))
        self.transcript_speak = self._wrap_non_tensor(self._unwrap_non_tensor(batch.get("transcript_speak", None)))
        self.transcript_hear = self._wrap_non_tensor(self._unwrap_non_tensor(batch.get("transcript_hear", None)))

        dtypes = dict()
        if self.video is not None:
            dtypes["video"] = self.video.dtype
        if self.audio_speak is not None:
            dtypes["audio_speak"] = self.audio_speak.dtype
        if self.audio_hear is not None:
            dtypes["audio_hear"] = self.audio_hear.dtype
        if self.key_press is not None and isinstance(self.key_press, torch.Tensor):
            dtypes["key_press"] = self.key_press.dtype
        if self.mouse_movement is not None and isinstance(self.mouse_movement, torch.Tensor):
            dtypes["mouse_movement"] = self.mouse_movement.dtype

        self.original_dtypes = self._wrap_non_tensor(dtypes)

    def to_dict(self):
        return {
            "video": self.video,
            "audio_speak": self.audio_speak,
            "audio_hear": self.audio_hear,
            "key_press": self.key_press,
            "mouse_movement": self.mouse_movement,
            "metadata": self._unwrap_non_tensor(self.metadata),
            "dataframe_indices": self.dataframe_indices,
            "transcript_speak": self._unwrap_non_tensor(self.transcript_speak),
            "transcript_hear": self._unwrap_non_tensor(self.transcript_hear),
        }

    def to_original_dtype_(self):
        dtype_map = self._unwrap_non_tensor(self.original_dtypes) or {}

        if self.video is not None and "video" in dtype_map:
            self.video = self.video.to(dtype_map["video"])
        if self.audio_speak is not None and "audio_speak" in dtype_map:
            self.audio_speak = self.audio_speak.to(dtype_map["audio_speak"])
        if self.audio_hear is not None and "audio_hear" in dtype_map:
            self.audio_hear = self.audio_hear.to(dtype_map["audio_hear"])
        if self.key_press is not None and "key_press" in dtype_map:
            self.key_press = self.key_press.to(dtype_map["key_press"])
        if self.mouse_movement is not None and "mouse_movement" in dtype_map:
            self.mouse_movement = self.mouse_movement.to(dtype_map["mouse_movement"])

    def to(self, device=None, dtype=None):
        """Move all tensors to device."""
        video = self.video.to(device=device, dtype=dtype) if self.video is not None else None
        audio_speak = self.audio_speak.to(device=device, dtype=dtype) if self.audio_speak is not None else None
        audio_hear = self.audio_hear.to(device=device, dtype=dtype) if self.audio_hear is not None else None
        key_press = self.key_press.to(device=device, dtype=dtype) if self.key_press is not None else None
        mouse_movement = self.mouse_movement.to(device=device, dtype=dtype) if self.mouse_movement is not None else None
        dataframe_indices = (
            self.dataframe_indices.to(device=device, dtype=torch.long)
            if self.dataframe_indices is not None
            else None
        )

        return FullData(
            batch={
                "video": video,
                "audio_speak": audio_speak,
                "audio_hear": audio_hear,
                "key_press": key_press,
                "mouse_movement": mouse_movement,
                "metadata": self._unwrap_non_tensor(self.metadata),
                "dataframe_indices": dataframe_indices,
                "transcript_speak": self._unwrap_non_tensor(self.transcript_speak),
                "transcript_hear": self._unwrap_non_tensor(self.transcript_hear),
            }
        )

    def load(self, prefix):
        return TensorDict.load_memmap(prefix)

    def save(self, index, prefix):
        self[index].memmap(prefix)

    def _save_video(self, index, path):
        np.save(path, self.video[index].cpu().numpy())

    def _save_audio_speak(self, index, path):
        np.save(path, self.audio_speak[index].cpu().numpy())

    def _save_audio_hear(self, index, path):
        np.save(path, self.audio_hear[index].cpu().numpy())

    def _save_key_press(self, index, path):
        np.save(path, self.key_press[index].cpu().numpy())

    def _save_mouse_movement(self, index, path):
        np.save(path, self.mouse_movement[index].cpu().numpy())

    @property
    def shapes(self):
        return {
            "video": self.video.shape if self.video is not None else None,
            "audio_speak": self.audio_speak.shape if self.audio_speak is not None else None,
            "audio_hear": self.audio_hear.shape if self.audio_hear is not None else None,
            "key_press": self.key_press.shape if self.key_press is not None else None,
            "mouse_movement": self.mouse_movement.shape if self.mouse_movement is not None else None,
            "transcript_speak": self._unwrap_non_tensor(self.transcript_speak),
            "transcript_hear": self._unwrap_non_tensor(self.transcript_hear),
        }

    @property
    def device(self):
        """Returns the device of the first valid tensor found in this FullData."""
        for field in ["video", "audio_speak", "audio_hear", "key_press", "mouse_movement", "dataframe_indices"]:
            val = getattr(self, field, None)
            if torch.is_tensor(val):
                return val.device
        return torch.device("cpu")

    def randn_like(self):
        batch = {}
        if self.video is not None:
            batch["video"] = torch.randn_like(self.video)
        if self.audio_speak is not None:
            batch["audio_speak"] = torch.randn_like(self.audio_speak)
        if self.audio_hear is not None:
            batch["audio_hear"] = torch.randn_like(self.audio_hear)
        if self.key_press is not None:
            batch["key_press"] = torch.randn_like(self.key_press)
        if self.mouse_movement is not None:
            batch["mouse_movement"] = torch.randn_like(self.mouse_movement)
        batch["transcript_speak"] = self._unwrap_non_tensor(self.transcript_speak)
        batch["transcript_hear"] = self._unwrap_non_tensor(self.transcript_hear)
        return FullData(batch=batch).to(self.device)

    def get_modality(self, modality: str) -> torch.Tensor:
        """Get a modality tensor by name."""
        if modality not in ["video", "audio_speak", "audio_hear", "key_press", "mouse_movement"]:
            raise ValueError(f"Unknown modality: {modality}")
        return getattr(self, modality)

    def set_modality(self, modality: str, value: torch.Tensor) -> None:
        """Set a modality tensor by name."""
        if modality not in ["video", "audio_speak", "audio_hear", "key_press", "mouse_movement"]:
            raise ValueError(f"Unknown modality: {modality}")
        setattr(self, modality, value)

    @staticmethod
    def slice_time(fd: "FullData", start: int, end: int) -> "FullData":
        """Slice all temporal tensors in a FullData object along dim=1."""
        if not isinstance(fd, FullData):
            raise TypeError("FullData.slice_time expects a FullData instance.")

        batch = fd.to_dict()
        for key in [
            "video",
            "audio_speak",
            "audio_hear",
            "key_press",
            "mouse_movement",
            "dataframe_indices",
        ]:
            tensor = batch.get(key)
            if torch.is_tensor(tensor):
                batch[key] = tensor[:, start:end]

        return FullData(batch=batch)

    @staticmethod
    def cat_time(fd_list: list["FullData"]) -> "FullData":
        """Concatenate a list of FullData objects along time dim=1."""
        if not fd_list:
            raise ValueError("FullData.cat_time requires a non-empty list.")
        if not all(isinstance(fd, FullData) for fd in fd_list):
            raise TypeError("FullData.cat_time expects a list of FullData objects.")

        first = fd_list[0]
        batch = first.to_dict()

        for modality in ["video", "audio_speak", "audio_hear", "key_press", "mouse_movement"]:
            tensors = [fd.get_modality(modality) for fd in fd_list]
            if all(t is None for t in tensors):
                continue
            if any(t is None for t in tensors):
                raise ValueError(f"Cannot concatenate modality '{modality}' when some entries are None.")
            batch[modality] = torch.cat(tensors, dim=1)

        key = "dataframe_indices"
        tensors = [getattr(fd, key, None) for fd in fd_list]
        if not all(t is None for t in tensors):
            if any(t is None for t in tensors):
                raise ValueError(f"Cannot concatenate '{key}' when some entries are None.")
            batch[key] = torch.cat(tensors, dim=1)

        return FullData(batch=batch)

    @staticmethod
    def cat_batch(fd_list: list["FullData"]) -> "FullData":
        """Concatenate a list of FullData objects along batch dim=0."""
        if not fd_list:
            raise ValueError("FullData.cat_batch requires a non-empty list.")
        if not all(isinstance(fd, FullData) for fd in fd_list):
            raise TypeError("FullData.cat_batch expects a list of FullData objects.")

        first = fd_list[0]
        batch = first.to_dict()

        for modality in ["video", "audio_speak", "audio_hear", "key_press", "mouse_movement"]:
            tensors = [fd.get_modality(modality) for fd in fd_list]
            if all(t is None for t in tensors):
                batch[modality] = None
                continue
            if any(t is None for t in tensors):
                raise ValueError(f"Cannot concatenate modality '{modality}' when some entries are None.")
            batch[modality] = torch.cat(tensors, dim=0)

        key = "dataframe_indices"
        tensors = [getattr(fd, key, None) for fd in fd_list]
        if all(t is None for t in tensors):
            batch[key] = None
        elif any(t is None for t in tensors):
            raise ValueError(f"Cannot concatenate '{key}' when some entries are None.")
        else:
            batch[key] = torch.cat(tensors, dim=0)

        for key in ["metadata", "transcript_speak", "transcript_hear"]:
            values = [d.get(key) for d in [fd.to_dict() for fd in fd_list]]
            if all(v is None for v in values):
                batch[key] = None
            elif values and all(isinstance(v, list) and len(v) == 1 for v in values):
                batch[key] = [v[0] for v in values]
            else:
                batch[key] = values

        return FullData(batch=batch)

    @staticmethod
    def infer_time_length(fd: "FullData") -> int:
        """Infer sequence length from the first available temporal tensor."""
        for modality in ["video", "audio_speak", "audio_hear", "key_press", "mouse_movement"]:
            tensor = fd.get_modality(modality)
            if tensor is not None:
                return int(tensor.shape[1])
        if fd.dataframe_indices is not None:
            return int(fd.dataframe_indices.shape[1])
        raise ValueError("Unable to infer time length: no temporal tensors are available.")

    @staticmethod
    def infer_batch_size(fd: "FullData") -> int:
        """Infer batch size from the first available temporal tensor."""
        for modality in ["video", "audio_speak", "audio_hear", "key_press", "mouse_movement"]:
            tensor = fd.get_modality(modality)
            if tensor is not None:
                return int(tensor.shape[0])
        if fd.dataframe_indices is not None:
            return int(fd.dataframe_indices.shape[0])
        raise ValueError("Unable to infer batch size: no temporal tensors are available.")


