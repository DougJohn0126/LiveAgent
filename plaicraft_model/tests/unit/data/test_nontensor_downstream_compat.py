import torch

from data.data_classes import FullData


def test_to_dict_unwraps_nontensor_fields_for_downstream():
    batch = {
        "video": torch.zeros(2, 4, 8, 8, dtype=torch.float16),
        "metadata": [[{"player_name": "p1", "player_gender": "f", "player_skill_level": "novice"}]],
        "transcript_speak": [[], [("hello", 10, 20)]],
        "transcript_hear": [[], []],
        "padding_mask": torch.ones(2, dtype=torch.bool),
    }

    full_data = FullData(batch=batch)
    payload = full_data.to_dict()

    assert isinstance(payload["metadata"], list)
    assert isinstance(payload["transcript_speak"], list)
    assert isinstance(payload["transcript_hear"], list)
    assert payload["metadata"] == batch["metadata"]


def test_to_original_dtype_works_with_wrapped_dtype_map():
    full_data = FullData(
        batch={
            "video": torch.zeros(1, 2, 4, 4, dtype=torch.float16),
            "padding_mask": torch.ones(1, dtype=torch.bool),
        }
    )

    full_data.video = full_data.video.float()
    assert full_data.video.dtype == torch.float32

    full_data.to_original_dtype_()
    assert full_data.video.dtype == torch.float16
