from pathlib import Path

from omegaconf import OmegaConf


def test_callback_targets_use_split_modules() -> None:
    expected_targets = {
        "async_semantic_evaluation.yaml": "src.utils.callbacks.async_semantic_evaluation.AsyncSemanticEvaluation",
        "module_step_time_profiler.yaml": "src.utils.callbacks.module_step_time_profiler.ModuleStepTimeProfiler",
        "peak_gpu_memory.yaml": "src.utils.callbacks.peak_gpu_memory.PeakGpuMemory",
        "rank_info.yaml": "src.utils.callbacks.rank_info.RankInfo",
        "sync_semantic_evaluation.yaml": "src.utils.callbacks.sync_semantic_evaluation.SyncSemanticEvaluation",
    }

    callbacks_dir = Path("configs") / "callbacks"
    for file_name, expected_target in expected_targets.items():
        cfg = OmegaConf.load(callbacks_dir / file_name)
        root_key = next(iter(cfg.keys()))
        assert cfg[root_key]["_target_"] == expected_target
