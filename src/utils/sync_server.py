#!/usr/bin/env python3
"""
sync_carya.py

Minimal rsync wrapper with hardcoded (easy-to-edit) settings.

Default behavior (when you run this file):
  - Sync FROM Carya -> Local
  - Uses update mode (-u)

How to use:
  python sync_carya.py

To change behavior:
  - Edit MODE to "pull", "push", or "delete"
  - Edit PULL_SYNC_PAIRS / PUSH_SYNC_PAIRS / DELETE_SYNC_PAIRS to add/remove
    (local_path, remote_path) pairs for each direction
  - Toggle UPDATE / DRY_RUN
  - Edit EXCLUDES to skip folders/files (e.g., .git/, __pycache__/)
"""

from __future__ import annotations

import subprocess

PathPair = tuple[str, str]


# -----------------------
# EASY-TO-EDIT SETTINGS
# -----------------------
REMOTE_HOST = "carya"

# Each entry is a (local_absolute_path, remote_absolute_path) pair.
# Add/remove pairs freely - all are synced in sequence.
PUSH_SYNC_PAIRS: list[PathPair] = [
    (
        "/Volumes/Works/Projects/MultiModalSummarizer/input",
        "/project/subhlok/dipayan/MultiModalSummarizer/input",
    ),
]

PULL_SYNC_PAIRS: list[PathPair] = [
    (
        "/Volumes/Works/LearningHub/vlm-forge/wandb",
        "/project/subhlok/dipayan/vlm-forge/wandb",
    ),
    (
        "/Volumes/Works/LearningHub/vlm-forge/outputs",
        "/project/subhlok/dipayan/vlm-forge/outputs",
    ),
]

# Pairs used for delete-sync (local -> server with --delete).
# Mirrors PUSH_SYNC_PAIRS by default; narrow this down if needed.
DELETE_SYNC_PAIRS: list[PathPair] = [
    (
        "/Volumes/Works/LearningHub/vlm-forge/wandb",
        "/project/subhlok/dipayan/vlm-forge/wandb",
    ),
]

# Default: pull (server -> local)
MODE = "pull"  # "pull", "push", or "delete"

# Default: update mode ON
UPDATE = True     # adds -u (skip overwriting newer dest files)
DRY_RUN = False   # adds --dry-run (preview only)

# Exclude noisy/dev artifacts (prevents copying back and forth)
EXCLUDES = [
    ".git/",
    ".DS_Store",
    "__pycache__/",
    "*.pyc",
    ".vscode/",
    ".cache/",
    "venv312/",
    ".env",
    "*.sh",
    "*obj*.o*",
    "*.mp4",
]


# -----------------------
# INTERNAL HELPERS
# -----------------------
def _slash(p: str) -> str:
    """Ensure trailing slash so rsync syncs directory contents."""
    return p if p.endswith("/") else p + "/"


def _run(cmd: list[str]) -> None:
    print("Running:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _add_excludes(cmd: list[str]) -> None:
    for pat in EXCLUDES:
        cmd.extend(["--exclude", pat])


def _build_cmd(src: str, dest: str, update: bool, dry_run: bool) -> list[str]:
    cmd = ["rsync", "-avP"]
    _add_excludes(cmd)
    if update:
        cmd.append("-u")
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend([_slash(src), _slash(dest)])
    return cmd


# -----------------------
# SYNC FUNCTIONS
# -----------------------
def sync_from_server(
    pairs: list[PathPair] = PULL_SYNC_PAIRS,
    update: bool = UPDATE,
    dry_run: bool = DRY_RUN,
) -> None:
    """Remote -> Local  (for every pair in `pairs`)."""
    for local_path, remote_path in pairs:
        src = f"{REMOTE_HOST}:{remote_path}"
        dest = local_path
        _run(_build_cmd(src, dest, update, dry_run))


def sync_to_server(
    pairs: list[PathPair] = PUSH_SYNC_PAIRS,
    update: bool = UPDATE,
    dry_run: bool = DRY_RUN,
) -> None:
    """Local -> Remote  (for every pair in `pairs`)."""
    for local_path, remote_path in pairs:
        src = local_path
        dest = f"{REMOTE_HOST}:{remote_path}"
        _run(_build_cmd(src, dest, update, dry_run))


def sync_delete_to_server(
    pairs: list[PathPair] = DELETE_SYNC_PAIRS,
    update: bool = UPDATE,
) -> None:
    """Local -> Remote with --delete: removes server files deleted locally.

    Always runs a dry-run preview first, then requires typing 'yes' to proceed.
    """
    print("\n=== DELETE SYNC: LOCAL → SERVER (--delete) ===")
    print("Files present on the server but missing locally will be DELETED on the server.")
    print("\n--- DRY RUN PREVIEW ---")
    for local_path, remote_path in pairs:
        src = local_path
        dest = f"{REMOTE_HOST}:{remote_path}"
        dry_cmd = _build_cmd(src, dest, update, dry_run=True)
        dry_cmd.append("--delete")
        _run(dry_cmd)

    print()
    response = input("Type 'yes' to execute the delete sync on the server, any other input cancels: ")
    if response.strip() != "yes":
        print("\nCancelled. No files were changed on the server.\n")
        return

    print("\n--- EXECUTING ---")
    for local_path, remote_path in pairs:
        src = local_path
        dest = f"{REMOTE_HOST}:{remote_path}"
        cmd = _build_cmd(src, dest, update, dry_run=False)
        cmd.append("--delete")
        _run(cmd)

    print("\n=== DELETE SYNC DONE ===\n")


# -----------------------
# DEFAULT ENTRY POINT
# -----------------------
if __name__ == "__main__":
    if MODE == "pull":
        sync_from_server()
    elif MODE == "push":
        sync_to_server()
    elif MODE == "delete":
        sync_delete_to_server()
    else:
        raise ValueError('MODE must be "pull", "push", or "delete"')
