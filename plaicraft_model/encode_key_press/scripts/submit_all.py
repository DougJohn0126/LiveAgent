#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from typing import List, Tuple
import json

# *** Configuration Starts Here ***
ERROR_MESSAGES = [
    "ValueError: min() arg is an empty sequence",
    "IndexError: list index out of range",
    "RuntimeError: KeyPress encoding failed for session",
    "FileNotFoundError: Session database not found",
    "sqlite3.OperationalError",
    "ValueError: No 'keyboard_events' found in the session data"
]
# *** Configuration Ends Here ***



def get_all_processed_tokens(processed_dir: Path) -> List[Tuple[str, str]]:
    """
    Retrieve all tokens from the 'processed' directory.
    """
    tokens = []
    for hashed_email_dir in processed_dir.iterdir():
        if not hashed_email_dir.is_dir():
            continue
        for token_dir in hashed_email_dir.iterdir():
            if token_dir.is_dir():
                tokens.append((hashed_email_dir.name, token_dir.name))
    return tokens


def write_tokens_to_file(tokens: List[Tuple[str, str]], tokens_dir: Path, prefix: str) -> str:
    """
    Write tokens to a uniquely named file in tokens_dir, one per line, formatted as hashedEmail/token.
    """
    try:
        timestamp = int(time.time())
        unique_id = uuid.uuid4().hex
        tokens_file = tokens_dir / f'{prefix}_tokens_{timestamp}_{unique_id}.txt'
        with open(tokens_file, 'w') as f:
            for hashed_email, token in tokens:
                f.write(f"{hashed_email}/{token}\n")
        return str(tokens_file.resolve())
    except Exception as e:
        print(f"[ERROR] Failed to write tokens to file: {e}")
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Submit key press encoding jobs for tokens.")
    parser.add_argument('--parent-dir', type=str, default='/ubc/cs/research/plai-scratch/plaicraft-dataset',
                        help='Parent directory containing session data.')
    parser.add_argument('--max-concurrent-jobs', type=int, default=8,
                        help='Maximum number of concurrent jobs.')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--reprocess-all', action='store_true',
                       help='Reprocess all tokens, ignoring their current processing status.')
    group.add_argument('--reprocess-only-token', type=str,
                       help=("Comma-separated list of tokens to reprocess. "
                             "Each token can be in the format 'hashedEmail/token' or just 'token'. "
                             "Examples: 'hashedEmail1/token1,hashedEmail2/token2', 'token3'"))
    group.add_argument('--reprocess-only-player', type=str,
                       help=("Comma-separated list of hashedEmail identifiers to reprocess all tokens belonging to these players. "
                             "Example: 'hashedEmail1,hashedEmail2'"))
    group.add_argument('--process-only-player', type=str,
                       help='Process only tokens belonging to the specified hashedEmail, following normal status logic.')
    
    parser.add_argument('--local-test', action='store_true',
                        help='Enable local testing mode without SLURM.')
    
    # ** New Argument for Model Checkpoint **
    parser.add_argument('--model-checkpoint', type=str, default="/ubc/cs/research/ubc_ml/plaicraft/plaicraft-data-preprocessing/encode_key_press/checkpoints/keyencoder_16_5_best_checkpoint.pt",
                        help='Path to the model checkpoint file.')
    
    return parser.parse_args()


def run_local_keypress_encoding(hashed_email: str, token: str, parent_dir: Path, model_checkpoint: str):
    """
    Run the key press encoding job locally.
    """
    try:
        # Define paths based on hashed_email and token
        processed_dir = parent_dir / 'processed' / hashed_email / token
        logs_dir = parent_dir / 'logs' / hashed_email / token
        status_dir = parent_dir / 'status' / hashed_email / token

        # Create necessary directories
        logs_dir.mkdir(parents=True, exist_ok=True)
        status_dir.mkdir(parents=True, exist_ok=True)

        # Print the logs directory path
        print(f"[INFO] Logs directory for token '{hashed_email}/{token}': {logs_dir.resolve()}")

        # Initialize status file to 'processing'
        encode_keypress_status_file = status_dir / "encode_keypress.status"
        encode_keypress_status_file.write_text("processing")

        # Define the correct session database path
        session_db_path = parent_dir / 'processed' / hashed_email / f"{token}.db"

        # Verify if the session database exists
        if not session_db_path.exists():
            raise FileNotFoundError(f"[ERROR] Session database not found: {session_db_path}")

        # Define log file paths
        encode_keypress_out = logs_dir / 'encode_keypress_local.out'
        encode_keypress_err = logs_dir / 'encode_keypress_local.err'

        # Run encode_keypress.py
        with open(encode_keypress_out, 'w') as fout, open(encode_keypress_err, 'w') as ferr:
            proc = subprocess.run([
                'python',
                '/ubc/cs/research/ubc_ml/plaicraft/plaicraft-data-preprocessing/encode_keypress/main.py',
                '--session_db_path', str(session_db_path),
                '--model_checkpoint', model_checkpoint,
                '--batch_size', '256',
                '--device', 'cuda',
                '--verbose'
            ], stdout=fout, stderr=ferr)

        if proc.returncode == 0:
            # Update status to 'success'
            encode_keypress_status_file.write_text("success")
            print(f"[INFO] Successfully encoded key presses for token '{hashed_email}/{token}'.")
        else:
            # Update status to 'failed'
            encode_keypress_status_file.write_text("failed")
            print(f"[ERROR] encode_keypress.py failed for token '{hashed_email}/{token}'. Check logs at {logs_dir.resolve()}.")

    except FileNotFoundError as e:
        print(e)
        print(f"[ERROR] Skipping processing for token '{hashed_email}/{token}' due to missing session database.")
    except Exception as e:
        print(f"[ERROR] Unexpected error when encoding key presses for token '{hashed_email}/{token}': {e}")


def submit_local_encode_keypress_jobs(tokens: List[Tuple[str, str]], parent_dir: Path, max_concurrent_jobs: int, model_checkpoint: str):
    """
    Submit key press encoding jobs to run locally with concurrency control.
    """
    job_count = 0
    with ThreadPoolExecutor(max_workers=max_concurrent_jobs) as executor:
        # Submit all jobs to the executor
        futures = [executor.submit(run_local_keypress_encoding, hashed_email, token, parent_dir, model_checkpoint) 
                   for hashed_email, token in tokens]

        for future in as_completed(futures):
            try:
                future.result()
                job_count += 1
            except Exception as e:
                print(f"[ERROR] Job generated an exception: {e}")

    print(f"[INFO] Total number of local key press encoding jobs submitted: {job_count}")


def has_specific_error(logs_dir: Path, error_messages: List[str]) -> bool:
    """
    Check if the latest encode_keypress*.err file in logs_dir contains any specified error messages.
    """
    err_files = sorted(logs_dir.glob("encode_keypress*.err"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not err_files:
        return False
    latest_err_file = err_files[0]
    try:
        content = latest_err_file.read_text()
        for error_message in error_messages:
            if error_message in content:
                return True
        return False
    except Exception as e:
        print(f"[ERROR] Could not read {latest_err_file}: {e}")
        return False


def token_needs_processing(status_dir: Path, hashed_email: str, token: str) -> bool:
    """
    Determine if a token needs processing based on the status files.
    Encode KeyPress can only be started if main.status is 'success'.
    """
    main_status_file = status_dir / hashed_email / token / "main.status"
    encode_keypress_status_file = status_dir / hashed_email / token / "encode_keypress.status"

    if not main_status_file.exists():
        return False  # Cannot proceed without main.status
    main_status = main_status_file.read_text().strip()
    if main_status != "success":
        return False  # Only proceed if main.status is 'success'

    if not encode_keypress_status_file.exists():
        return True  # No encode_keypress.status means it hasn't been processed yet
    encode_status = encode_keypress_status_file.read_text().strip()
    return encode_status != "success"


def submit_job_array_in_batches_keypress(tokens_file: str, parent_dir: str, max_concurrent_jobs: int, model_checkpoint: str, max_array_size: int = 1000):
    """
    Submit SLURM job arrays in batches for key press encoding, each not exceeding max_array_size.
    Each batch is submitted after the previous one finishes, regardless of success.
    """
    try:
        # Ensure the tokens file exists and is not empty
        if not os.path.exists(tokens_file):
            raise FileNotFoundError(f"[ERROR] Tokens file not found: {tokens_file}")

        with open(tokens_file, 'r') as f:
            tokens = f.readlines()

        num_tokens = len(tokens)
        if num_tokens == 0:
            print("[INFO] No tokens to submit. The tokens file is empty.")
            return

        # Determine the number of batches needed
        batches = [tokens[i:i + max_array_size] for i in range(0, num_tokens, max_array_size)]
        total_batches = len(batches)
        print(f"[INFO] Total tokens: {num_tokens}. Submitting in {total_batches} batches.")

        # Path to the SLURM script
        current_script_path = Path(__file__).resolve().parent
        sbatch_script_path = current_script_path / "sbatch_encode_keypress.sh"  # Ensure correct script name

        # Check if the SLURM script exists
        if not sbatch_script_path.exists():
            raise FileNotFoundError(f"[ERROR] SLURM script not found: {sbatch_script_path}")

        # Create a temporary file for each batch
        tokens_dir = Path(tokens_file).parent  # Assuming tokens_file is in tokens_dir
        batch_job_ids = []
        for batch_num, batch in enumerate(batches, start=1):
            # Write batch tokens to a temporary file
            batch_tokens_file = tokens_dir / f'encode_keypress_tokens_batch{batch_num}.txt'
            with open(batch_tokens_file, 'w') as bf:
                for token in batch:
                    bf.write(token)

            # Construct sbatch command with proper array job limit
            sbatch_command = [
                'sbatch',
          #      "--test-only",
                '--array=1-{}%{}'.format(len(batch), max_concurrent_jobs),  # Corrected array specification
                '--export=ALL,TOKENS_FILE={},PARENT_DIR={},MODEL_CHECKPOINT={}'.format(
                    batch_tokens_file, parent_dir, model_checkpoint
                ),
                str(sbatch_script_path)
            ]

            # If not the first batch, add dependency
            if batch_num > 1 and batch_job_ids:
                dependency = f"afterany:{batch_job_ids[-1]}"
                sbatch_command.insert(1, f'--dependency={dependency}')

            # Submit the job
            result = subprocess.run(sbatch_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"[INFO] Batch {batch_num} submitted successfully: {result.stdout.strip()}")

            # Extract the job ID from SLURM's output
            # Example SLURM output: "Submitted batch job 123456"
            if result.stdout:
                job_id = result.stdout.strip().split()[-1]
                batch_job_ids.append(job_id)
                print(f"[INFO] Batch {batch_num} assigned Job ID: {job_id}")
            else:
                raise ValueError(f"No job ID returned for batch {batch_num}.")

        print(f"[INFO] All {total_batches} batches submitted successfully.")
        return batch_job_ids
    except FileNotFoundError as e:
        print(e)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to submit job array: {e.stderr.strip()}")
    except Exception as e:
        print(f"[ERROR] Unexpected error when submitting job array: {e}")


def main_submission():
    """
    Main function to orchestrate key press encoding job submissions.
    """
    args = parse_arguments()
    parent_dir = Path(args.parent_dir)
    max_concurrent_jobs = args.max_concurrent_jobs
    reprocess_all = args.reprocess_all
    reprocess_only_token = args.reprocess_only_token
    reprocess_only_player = args.reprocess_only_player
    process_only_player = args.process_only_player
    local_test = args.local_test
    model_checkpoint = args.model_checkpoint  # Get from command-line argument

    # Validate parent directory
    if not parent_dir.exists():
        print(f"[ERROR] Parent directory does not exist: {parent_dir}")
        sys.exit(1)

    processed_dir = parent_dir / 'processed'
    status_dir = parent_dir / 'status'

    # Validate model checkpoint
    model_checkpoint_path = Path(model_checkpoint)
    if not model_checkpoint_path.is_file():
        print(f"[ERROR] Model checkpoint not found: {model_checkpoint}")
        sys.exit(1)

    # Retrieve all tokens
    all_tokens = get_all_processed_tokens(processed_dir)
    total_all_tokens = len(all_tokens)

    # Initialize counters
    skipped_due_to_main_status = 0
    skipped_due_to_encode_success = 0

    # Determine which tokens to process
    if reprocess_all:
        tokens_to_process = all_tokens
        print(f"[INFO] --reprocess-all specified. Reprocessing all {len(tokens_to_process)} tokens.")
    elif reprocess_only_token:
        tokens_to_process_set = set()
        requested_items = [s.strip() for s in reprocess_only_token.split(',') if s.strip()]
        for item in requested_items:
            if '/' in item:
                try:
                    hashed_email, token_name = item.split('/', 1)
                    if (hashed_email, token_name) in all_tokens:
                        tokens_to_process_set.add((hashed_email, token_name))
                        print(f"[INFO] Added '{hashed_email}/{token_name}' for reprocessing.")
                    else:
                        print(f"[WARNING] Token '{item}' does not exist. Skipping.")
                except ValueError:
                    print(f"[WARNING] Invalid token format '{item}'. Expected 'hashedEmail/token'. Skipping.")
            else:
                # Assume it's a token
                tokens_matching_token = [t for t in all_tokens if t[1] == item]
                if tokens_matching_token:
                    tokens_to_process_set.update(tokens_matching_token)
                    for he, tk in tokens_matching_token:
                        print(f"[INFO] Added token '{he}/{tk}' for reprocessing.")
                else:
                    print(f"[WARNING] Token '{item}' does not exist. Skipping.")

        tokens_to_process = list(tokens_to_process_set)
        print(f"[INFO] --reprocess-only-token specified. Preparing to reprocess {len(tokens_to_process)} tokens.")
    elif reprocess_only_player:
        tokens_to_process_set = set()
        requested_players = [s.strip() for s in reprocess_only_player.split(',') if s.strip()]
        for player in requested_players:
            player_tokens = [t for t in all_tokens if t[0] == player]
            if player_tokens:
                tokens_to_process_set.update(player_tokens)
                print(f"[INFO] Added all tokens for player '{player}' for reprocessing ({len(player_tokens)} tokens).")
            else:
                print(f"[WARNING] Player '{player}' does not exist or has no tokens. Skipping.")

        tokens_to_process = list(tokens_to_process_set)
        print(f"[INFO] --reprocess-only-player specified. Preparing to reprocess {len(tokens_to_process)} tokens.")
    elif process_only_player:
        # Process only tokens for that user following status rules (like default mode)
        user_tokens = [t for t in all_tokens if t[0] == process_only_player]
        if not user_tokens:
            print(f"[WARNING] No tokens found for user '{process_only_player}'. Nothing to process.")
            sys.exit(0)
        
        # Filter by status, only include tokens needing processing
        tokens_to_process = [t for t in user_tokens if token_needs_processing(status_dir, t[0], t[1])]
        skipped_due_to_main_status = len(user_tokens) - len(tokens_to_process)
        print(f"[INFO] --process-only-player specified for '{process_only_player}'.")
        print(f"[INFO] Found {len(user_tokens)} tokens total for this user.")
        print(f"[INFO] {skipped_due_to_main_status} tokens are already processed successfully or main.status not 'success' and will be ignored.")
    else:
        # Default logic: process only tokens that need processing
        tokens_to_process = []
        for hashed_email, token in all_tokens:
            if token_needs_processing(status_dir, hashed_email, token):
                tokens_to_process.append((hashed_email, token))
            else:
                skipped_due_to_main_status += 1
        print(f"[INFO] Preparing to process {len(tokens_to_process)} tokens based on processing flags.")

    if not tokens_to_process and not (reprocess_only_token or reprocess_only_player or process_only_player):
        # If no tokens to process at all (and we are not in the user scenario that leads to exit)
        print("[INFO] No tokens require processing. Exiting.")
        sys.exit(0)

    # ------------------------------
    # Filter tokens by error logs:
    # If a token's latest error log contains one of the ERROR_MESSAGES,
    # mark its status as failed and do not include it for processing.
    # ------------------------------
    tokens_to_process_filtered = []
    ignored_due_to_error = 0
    for hashed_email, token in tokens_to_process:
        logs_dir = parent_dir / 'logs' / hashed_email / token
        if has_specific_error(logs_dir, ERROR_MESSAGES):
            ignored_due_to_error += 1
            # Set its status file to failed
            status_file = status_dir / hashed_email / token / "encode_keypress.status"
            status_file.parent.mkdir(parents=True, exist_ok=True)
            status_file.write_text("failed")
            print(f"[INFO] Token '{hashed_email}/{token}' ignored due to error, status set to 'failed'.")
        else:
            tokens_to_process_filtered.append((hashed_email, token))

    tokens_to_process = tokens_to_process_filtered

    total_ignored_jobs = skipped_due_to_main_status + ignored_due_to_error
    total_submitted = len(tokens_to_process)

    # Print final summary
    if process_only_player:
        print("\n[INFO] Summary for user '{}':".format(process_only_player))
        print(f" - Total tokens for user: {len([t for t in all_tokens if t[0] == process_only_player])}")
        print(f" - Ignored (already processed or main.status not 'success'): {skipped_due_to_main_status}")
        print(f" - Ignored due to known errors: {ignored_due_to_error}")
        print(f" - Will be submitted for processing: {total_submitted}")
    elif reprocess_only_token or reprocess_only_player:
        print("\n[INFO] Summary for reprocessing:")
        print(f" - Total tokens available: {total_all_tokens}")
        print(f" - Ignored (not selected for reprocessing or main.status not 'success'): {skipped_due_to_main_status}")
        print(f" - Ignored due to known errors: {ignored_due_to_error}")
        print(f" - Will be submitted for processing: {total_submitted}")
    else:
        # General summary
        print(f"\nTotal number of jobs submitted: {total_submitted}")
        print(f"Total number of ignored jobs: {total_ignored_jobs}")

    if not tokens_to_process and not (reprocess_only_token or reprocess_only_player or process_only_player):
        # After error filtering, nothing to process
        print("[INFO] No tokens remain to process after filtering errors. Exiting.")
        sys.exit(0)

    # Print logs directory paths for all tokens to be processed
    print("\n[INFO] Logs directories for tokens to be processed:")
    for hashed_email, token in tokens_to_process:
        logs_dir = parent_dir / 'logs' / hashed_email / token
        try:
            abs_path = logs_dir.resolve()
            print(f" - Token '{hashed_email}/{token}': {abs_path}")
        except Exception as e:
            print(f"[ERROR] Could not resolve path for 'logs/{hashed_email}/{token}': {e}")
    print()

    if local_test:
        print("[INFO] Running key press encoding jobs in local testing mode.")
        # Use the model_checkpoint passed as a command-line argument
        submit_local_encode_keypress_jobs(tokens_to_process, parent_dir, max_concurrent_jobs, model_checkpoint)
    else:
        print("[INFO] Submitting key press encoding jobs to SLURM.")
        # Create a dedicated directory for token files
        tokens_dir = parent_dir / 'tokens_submission_files'
        tokens_dir.mkdir(parents=True, exist_ok=True)

        # Write tokens to a uniquely named file
        tokens_file = write_tokens_to_file(tokens_to_process, tokens_dir, prefix="encode_keypress")

        # Ensure the provided model checkpoint exists (already checked above)
        # Submit the job array in batches with dependencies
        batch_job_ids = submit_job_array_in_batches_keypress(tokens_file, str(parent_dir), max_concurrent_jobs, model_checkpoint)
        return batch_job_ids
        # Optional: Implement cleanup logic here if needed


if __name__ == "__main__":
    batch_job_ids = main_submission()

    if batch_job_ids:
        print(f"The IDs in JSON array: {json.dumps(batch_job_ids)}")
    else:
        print(f"The IDs in JSON array: {json.dumps([])}")