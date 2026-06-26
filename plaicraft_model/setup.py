"""
Setup file for plaicraft package.

Package configuration is primarily in pyproject.toml.
This file provides backward compatibility for older tools and an optional
interactive bootstrap flow when executed directly:

    python setup.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
import venv

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def _prompt_yes_no(message: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{message} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer with 'y' or 'n'.")


def _run(cmd: list[str], step_name: str) -> bool:
    print(f"\n[{step_name}] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
        print(f"[{step_name}] Completed.")
        return True
    except subprocess.CalledProcessError as error:
        print(f"[{step_name}] Failed with exit code {error.returncode}.")
        return False


def _create_conda_env(env_name: str, python_version: str) -> Optional[str]:
    conda_executable = shutil.which("conda")
    if not conda_executable:
        print("conda command not found. Skipping conda environment creation.")
        return None

    create_cmd = [conda_executable, "create", "-y", "-n", env_name, f"python={python_version}"]
    if not _run(create_cmd, "Create conda env"):
        return None

    return env_name


def _create_venv(venv_path: Path) -> Optional[Path]:
    print(f"\n[Create venv] Creating virtual environment at {venv_path}")
    try:
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(str(venv_path))
    except Exception as error:  # pragma: no cover - defensive for local toolchain differences
        print(f"[Create venv] Failed: {error}")
        return None

    python_path = venv_path / "bin" / "python"
    if not python_path.exists():
        print(f"[Create venv] Could not find interpreter at {python_path}")
        return None

    print("[Create venv] Completed.")
    return python_path


def _choose_python_for_install() -> Tuple[list[str], str]:
    use_custom_env = _prompt_yes_no("Step 1 (optional): Create a virtual environment?", default=True)

    if not use_custom_env:
        print("[Step 1] Skipped. Using current Python environment.")
        return [sys.executable], "Current Python"

    env_type = input("Choose environment type ('conda' or 'venv') [conda]: ").strip().lower() or "conda"

    if env_type == "conda":
        env_name = input("Conda environment name [plaicraft]: ").strip() or "plaicraft"
        py_ver = input("Python version for conda env [3.10]: ").strip() or "3.10"
        created_env = _create_conda_env(env_name, py_ver)
        if created_env is None:
            return [sys.executable], "Current Python"

        conda_executable = shutil.which("conda")
        if not conda_executable:
            return [sys.executable], "Current Python"

        return [conda_executable, "run", "-n", created_env, "python"], f"Conda env '{created_env}'"

    if env_type == "venv":
        raw_name = input("venv directory name/path [.venv]: ").strip() or ".venv"
        venv_path = Path(raw_name)
        if not venv_path.is_absolute():
            venv_path = ROOT / venv_path
        python_path = _create_venv(venv_path)
        if python_path is None:
            return [sys.executable], "Current Python"
        return [str(python_path)], f"venv at '{venv_path}'"

    print("Unknown environment type. Skipping environment creation.")
    return [sys.executable], "Current Python"


def _download_hf_models(python_cmd: list[str]) -> None:
    if not _prompt_yes_no("Step 3 (optional): Download HuggingFace models now?", default=False):
        print("[Step 3] Skipped model download.")
        return

    _run(python_cmd + ["scripts/download_models.py"], "Download HF models")


def _fill_empty_env_values(env_path: Path) -> None:
    lines = env_path.read_text(encoding="utf-8").splitlines()
    updated_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith("#") or "=" not in line:
            updated_lines.append(line)
            continue

        key, value = line.split("=", 1)
        current_value = value.strip()

        if current_value in {'""', "''"}:
            user_value = input(f"Value for {key.strip()} (press Enter to keep empty): ")
            updated_lines.append(f'{key}="{user_value}"')
        else:
            updated_lines.append(line)

    env_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def _setup_env_file() -> None:
    if not _prompt_yes_no("Step 4 (optional): Create/update .env from .env.example?", default=True):
        print("[Step 4] Skipped .env setup.")
        return

    env_path = ROOT / ".env"
    env_example_path = ROOT / ".env.example"

    if not env_example_path.exists():
        print(".env.example not found. Skipping .env setup.")
        return

    if not env_path.exists():
        shutil.copyfile(env_example_path, env_path)
        print("Created .env from .env.example")
    else:
        print(".env already exists; only empty values will be prompted.")

    _fill_empty_env_values(env_path)
    print("[Step 4] .env setup complete.")


def run_bootstrap_flow() -> int:
    print("PLAICraft setup wizard")
    print("This will run optional setup steps, plus mandatory editable install.\n")

    python_cmd, python_label = _choose_python_for_install()
    print(f"\nUsing interpreter: {python_label}")

    if not _run(python_cmd + ["-m", "pip", "install", "-e", "."], "Step 2: pip install -e ."):
        print("Mandatory install step failed. Exiting.")
        return 1

    _download_hf_models(python_cmd)
    _setup_env_file()

    print("\nSetup flow finished.")
    return 0


if __name__ == "__main__" and (len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] == "bootstrap")):
    if len(sys.argv) > 1 and sys.argv[1] == "bootstrap":
        del sys.argv[1]
    raise SystemExit(run_bootstrap_flow())


setup(
    name="plaicraft",
    version="0.0.1",
    description="plaicraft agent model",
    author="",
    author_email="",
    url="https://github.com/plai-group/plaicraft-model-pi0",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
        ]
    },
)