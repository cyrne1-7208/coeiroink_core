from pathlib import Path

import pytest
import torch

from coeirocore.espnet_vits_loader import _load_inference_state


def test_load_inference_state_copies_checkpoint_storage(tmp_path: Path) -> None:
    source = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "model.pth"
    torch.save(source.state_dict(), checkpoint)
    expected = {name: value.clone() for name, value in source.state_dict().items()}

    destination = torch.nn.Linear(3, 2)
    _load_inference_state(destination, checkpoint)
    checkpoint.unlink()

    for name, value in destination.state_dict().items():
        assert torch.equal(value, expected[name])


def test_load_inference_state_rejects_missing_weight(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    torch.save({}, checkpoint)

    with pytest.raises(RuntimeError, match="missing inference weights"):
        _load_inference_state(torch.nn.Linear(3, 2), checkpoint)
