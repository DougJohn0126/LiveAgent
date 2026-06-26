from lightning.pytorch.callbacks import RichProgressBar


class SafeRichProgressBar(RichProgressBar):
    """Rich progress bar that no-ops if Lightning resume leaves bar ids uninitialized."""

    def _update(self, progress_bar_id, current, visible=True):
        if progress_bar_id is None:
            return
        return super()._update(progress_bar_id, current, visible=visible)
