"""Hugging Face Hub downloads, pinned to the revisions the defaults were measured with (#47).

A repo id downloads its pinned commit (HUB_REVISIONS), not whatever its main branch holds by then: a fresh start runs
the weights the README's numbers come from. ``repo@revision`` names another commit, branch or tag of any repo;
VOXCPM_HUB_LATEST=1 takes the main branch of the pinned repos too (to try a new upload before it is pinned).
"""

from __future__ import annotations

import os

# repo id -> the commit the defaults were benchmarked and checked with (a new upload: bump it with the numbers)
HUB_REVISIONS = {
    "VoxCPM-Kernel/VoxCPM2-Arabic-Dialects": "6b0845886d824f901ca19456fafdc1a1f13519d1",
    "VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT": "6f5c313dda5c28f60cfcad3591e549c66775e57c",
    "VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3": "a70fb905eb8135fbd49166535c1c17db988475db",
    "VoxCPM-Kernel/Arabic-Hindi-Reference-Voices": "0c209b3f4eb548bbcc732339fc9fda3086ea37d6",
}


def split_spec(spec: str) -> tuple[str, str | None, str | None]:
    """``repo[@revision][:file]`` -> (repo, revision, file); without @revision the repo's pinned one (None: its main
    branch, also with VOXCPM_HUB_LATEST=1)."""
    repo, _, filename = spec.partition(":")
    repo, at, revision = repo.partition("@")
    if not at:
        revision = None if os.environ.get("VOXCPM_HUB_LATEST") == "1" else HUB_REVISIONS.get(repo)
    return repo, revision or None, filename or None


def download(spec: str, **kwargs) -> str:
    """The local path of ``spec`` (``repo[@revision][:file]``): the repo's snapshot, or that one file of it; kwargs
    go to huggingface_hub (repo_type, allow_patterns, ...)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    repo, revision, filename = split_spec(spec)
    if filename is not None:
        return hf_hub_download(repo, filename, revision=revision, **kwargs)
    return snapshot_download(repo, revision=revision, **kwargs)
