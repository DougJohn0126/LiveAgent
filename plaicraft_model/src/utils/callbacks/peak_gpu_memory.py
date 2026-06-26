import lightning.pytorch as pl
import torch


class PeakGpuMemory(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize(pl_module.device)
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not torch.cuda.is_available():
            return
        dev = pl_module.device
        torch.cuda.synchronize(dev)
        peak_alloc = torch.cuda.max_memory_allocated(dev) / (1024**2)
        peak_resvd = torch.cuda.max_memory_reserved(dev) / (1024**2)
        pl_module.log(
            "mem/peak_allocated_mb",
            peak_alloc,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
        )
        pl_module.log(
            "mem/peak_reserved_mb",
            peak_resvd,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
        )
