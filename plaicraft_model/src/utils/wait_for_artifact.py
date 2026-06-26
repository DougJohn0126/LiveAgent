import argparse
import sys
import time
import wandb


def _valid_ref(ref: str) -> bool:
    # Minimal sanity: must include at least "entity/project/name"
    return ref.count("/") >= 2


def wait_for_artifact(artifact_ref: str, timeout_sec: int = 1200, poll_sec: float = 2.0, verbose: bool = True) -> bool:
    if not _valid_ref(artifact_ref):
        raise ValueError(f"Invalid artifact ref: {artifact_ref}")
    # If caller omitted an alias or version, use ':latest'
    if ":" not in artifact_ref:
        artifact_ref = f"{artifact_ref}:latest"

    api = wandb.Api()
    deadline = time.time() + timeout_sec

    attempt = 0
    last_err = None
    while time.time() < deadline:
        attempt += 1
        try:
            art = api.artifact(artifact_ref)
            _ = art.id
            if verbose:
                print(f"[artifact_gate] READY on attempt {attempt}: {art.name}:{art.version}")
            return True
        except Exception as e:
            last_err = e
            if verbose:
                print(f"[artifact_gate] not ready (attempt {attempt}): {e!s}")
            time.sleep(poll_sec)

    if verbose:
        print(f"[artifact_gate] TIMEOUT after {attempt} attempts. Last error: {last_err}")
    return False


def main():
    p = argparse.ArgumentParser(description="Block until a W&B artifact (alias or version) is resolvable.")
    p.add_argument("--artifact-ref", required=True, help="e.g. entity/project/model-<runid>-last:latest or with :vNN")
    p.add_argument("--timeout-sec", type=int, default=1200)
    p.add_argument("--poll-sec", type=float, default=2.0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    ok = wait_for_artifact(
        artifact_ref=args.artifact_ref,
        timeout_sec=args.timeout_sec,
        poll_sec=args.poll_sec,
        verbose=not args.quiet
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
