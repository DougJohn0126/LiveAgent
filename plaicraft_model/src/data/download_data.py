import subprocess
import logging
import argparse
import sys
from pathlib import Path

S3_BUCKET = 'plai-processed-data-bucket-prod'
LOCAL_DIR = "PATH_TO_LOCAL" # REMEMBER TO CHANGE THIS
S3_URI_BASE = f"s3://{S3_BUCKET}/"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_args(emails, sessions):
    if sessions and not emails:
        logger.error("ARGUMENT ERROR: You specified --sessions but did not specify --emails.")
        logger.error("To maintain search efficiency, you must provide the hashedEmail(s).")
        sys.exit(1)

def run_targeted_sync(s3_source, local_dest, data_format, dry_run, relative_wildcard):
    """
    Runs a specific sync command for a specific folder depth.
    relative_wildcard: 
      - "" (empty string) if syncing a specific session (files are at root of sync)
      - "*/" if syncing an email folder (files are one folder deep)
    """
    
    # 1. Base Command
    cmd = [
        "aws", "s3", "sync",
        s3_source,
        str(local_dest),
        "--exclude", "*", 
    ]

    # 2. Add Includes (Relative to the s3_source)
    p = relative_wildcard # Prefix for patterns
    
    # Base files (wav, db, json, mp4)
    for ext in ["*.wav", "*.db", "*.json", "*.mp4"]:
        cmd.extend(["--include", f"{p}{ext}"])

    # Format specific files
    if data_format == 'hdf5':
        # Sync the entire folder content for video hdf5
        cmd.extend(["--include", f"{p}encoded_video_hdf5/*"])
        # Sync specific files for audio
        cmd.extend(["--include", f"{p}encoded_audio_continuous/*.hdf5"])
        cmd.extend(["--include", f"{p}encoded_audio_discrete/*.hdf5"])

    elif data_format == 'pt':
        cmd.extend(["--include", f"{p}encoded_video/*"])
        cmd.extend(["--include", f"{p}encoded_audio_continuous/*.pt"])
        # encoded_audio_discrete is usually skipped for PT format

    if dry_run:
        cmd.append("--dryrun")

    # 3. Execution
    # Standard output is NOT captured, so it will stream directly to your console
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        logger.warning(f"Could not sync {s3_source}. It might not exist.")

def sync_s3_to_local(confirm_download: bool, emails: list, sessions: list, data_format: str):
    validate_args(emails, sessions)
    
    # Defaults
    email_list = emails if emails else []
    session_list = sessions if sessions else []
    is_dry_run = not confirm_download

    if not confirm_download:
        logger.warning("DRY RUN ENABLED. Run with --confirm-download to execute.")

    # SCENARIO 1: User provided Specific Emails
    if email_list:
        for email in email_list:
            
            # Sub-scenario A: Specific Sessions provided (FASTEST)
            if session_list:
                for sess in session_list:
                    s3_src = f"{S3_URI_BASE}{email}/{sess}/"
                    local_dst = Path(LOCAL_DIR) / email / sess
                    
                    logger.info(f"Targeting Session: {s3_src}")
                    # Prefix is empty because we are inside the session folder
                    run_targeted_sync(s3_src, local_dst, data_format, is_dry_run, relative_wildcard="")
            
            # Sub-scenario B: No Sessions provided (Sync whole email folder)
            else:
                s3_src = f"{S3_URI_BASE}{email}/"
                local_dst = Path(LOCAL_DIR) / email
                
                logger.info(f"Targeting Email Folder: {s3_src}")
                # Prefix is "*/" because files are inside subfolders (session_ids)
                run_targeted_sync(s3_src, local_dst, data_format, is_dry_run, relative_wildcard="*/")

    # SCENARIO 2: No filters provided (Global Scan - SLOW)
    else:
        logger.warning("No filters provided. Syncing ENTIRE bucket (This will take a while to start...)")
        run_targeted_sync(S3_URI_BASE, Path(LOCAL_DIR), data_format, is_dry_run, relative_wildcard="*/*/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-download", action="store_true", help="Execute download")
    parser.add_argument("--emails", nargs='+', default=[], help="List of hashedEmails")
    parser.add_argument("--sessions", nargs='+', default=[], help="List of session_ids")
    parser.add_argument("--format", choices=['pt', 'hdf5'], required=True, help="Data format version")

    args = parser.parse_args()

    sync_s3_to_local(args.confirm_download, args.emails, args.sessions, args.format)