import os

import lightning.pytorch as pl
import torch


class RankInfo(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        print(
            "[RANK INFO]:",
            "global_rank",
            trainer.global_rank,
            "local_rank",
            trainer.local_rank,
            "node_rank",
            trainer.node_rank,
            "world_size",
            trainer.world_size,
            "CUDA_VISIBLE_DEVICES",
            os.getenv("CUDA_VISIBLE_DEVICES"),
            "current_device",
            current_device,
        )
