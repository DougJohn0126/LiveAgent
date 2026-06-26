from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile


_ZIP_COMPRESS_LEVEL = 1


def zip_checkpoint_path(checkpoint_path: str, output_dir: str | None = None) -> str:
    path = Path(checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    base_dir = Path(output_dir) if output_dir else (Path(tempfile.gettempdir()) / "plaicraft_ckpt_zip")
    base_dir.mkdir(parents=True, exist_ok=True)

    zip_path = base_dir / f"{path.name}.zip"

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=_ZIP_COMPRESS_LEVEL,
    ) as zf:
        if path.is_file():
            zf.write(path, arcname=path.name)
        else:
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    arcname = str(Path(path.name) / file_path.relative_to(path))
                    zf.write(file_path, arcname=arcname)

    return str(zip_path)


def unzip_checkpoint_archive(archive_path: str, output_dir: str | None = None) -> str:
    archive = Path(archive_path).resolve()
    if not archive.exists() or archive.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Checkpoint archive does not exist or is not a .zip: {archive_path}")

    extract_dir = Path(output_dir) if output_dir else archive.with_suffix("")
    extract_dir.mkdir(parents=True, exist_ok=True)

    if not any(extract_dir.iterdir()):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(extract_dir)

    top_entries = [p for p in extract_dir.iterdir()]
    if len(top_entries) == 1:
        return str(top_entries[0].resolve())
    return str(extract_dir.resolve())
