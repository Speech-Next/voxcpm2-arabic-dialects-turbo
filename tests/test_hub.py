"""Hub downloads pinned to a revision (#47), and the CUDA kernels' build shared by every checkout of a source."""

import sys
import types

import pytest

from nanovllm_voxcpm import hub
from nanovllm_voxcpm.dialect_tts import DEFAULT_DIT, DEFAULT_MODEL, resolve


@pytest.fixture
def calls(monkeypatch):
    """huggingface_hub's two downloads, recorded instead of run."""
    seen = []
    fake = types.SimpleNamespace(
        snapshot_download=lambda repo, **kw: seen.append(("snapshot", repo, kw)) or f"/snap/{repo}",
        hf_hub_download=lambda repo, filename, **kw: seen.append(("file", repo, filename, kw)) or f"/file/{filename}",
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.delenv("VOXCPM_HUB_LATEST", raising=False)
    return seen


def test_the_defaults_download_their_pinned_revision(calls):
    from nanovllm_voxcpm.dialect_routing import DEFAULT_TAGGED_MODEL

    resolve(DEFAULT_MODEL)
    resolve(DEFAULT_DIT)
    repo, _, filename = DEFAULT_DIT.partition(":")
    assert calls == [("snapshot", DEFAULT_MODEL, {"revision": hub.HUB_REVISIONS[DEFAULT_MODEL]}),
                     ("file", repo, filename, {"revision": hub.HUB_REVISIONS[repo]})]
    assert DEFAULT_TAGGED_MODEL in hub.HUB_REVISIONS
    assert all(len(sha) == 40 for sha in hub.HUB_REVISIONS.values())


def test_a_revision_in_the_spec_and_latest(calls, monkeypatch):
    assert hub.split_spec("org/repo@v2:weights.pt") == ("org/repo", "v2", "weights.pt")
    assert hub.split_spec("org/repo") == ("org/repo", None, None)  # not pinned: its main branch
    assert hub.split_spec(f"{DEFAULT_MODEL}@main") == (DEFAULT_MODEL, "main", None)
    monkeypatch.setenv("VOXCPM_HUB_LATEST", "1")
    assert hub.split_spec(DEFAULT_MODEL) == (DEFAULT_MODEL, None, None)
    hub.download("org/data@abc", repo_type="dataset", allow_patterns=["*.wav"])
    assert calls[-1] == ("snapshot", "org/data", {"revision": "abc", "repo_type": "dataset",
                                                  "allow_patterns": ["*.wav"]})


def test_a_local_path_is_not_downloaded(calls, tmp_path):
    assert resolve(str(tmp_path)) == str(tmp_path)
    assert calls == []


def test_one_kernel_build_for_every_checkout_of_a_source(monkeypatch, tmp_path):
    """ninja's command line holds the source's path: the build compiles a copy under the build directory, the same
    path from every checkout (a start from another worktree compiled everything again, ~160 s)."""
    from nanovllm_voxcpm.kernels import cuda_ext

    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "ext"))
    paths = []
    for checkout in ("a", "b"):
        src = tmp_path / checkout / "resunit_fused.cu"
        src.parent.mkdir()
        src.write_text("// the same source in two checkouts\n")
        paths.append(cuda_ext._stable_source(str(src), "voxcpm_resunit_fused_sm86_0123456789"))
    assert paths[0] == paths[1] == str(tmp_path / "ext" / "voxcpm_resunit_fused_sm86_0123456789" / "resunit_fused.cu")
    assert open(paths[0]).read() == "// the same source in two checkouts\n"
