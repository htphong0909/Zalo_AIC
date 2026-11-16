"""Unit tests for Model Soup uniform weight averaging."""
import torch
from safetensors.torch import save_file, load_file
from scripts.merge_lfm3_soup import average_safetensors_weights


def test_average_safetensors_weights(tmp_path):
    dir1 = tmp_path / "ckpt1"
    dir2 = tmp_path / "ckpt2"
    dir1.mkdir()
    dir2.mkdir()

    # Create dummy tensors
    t1 = {"weight": torch.tensor([2.0, 4.0], dtype=torch.float32)}
    t2 = {"weight": torch.tensor([4.0, 8.0], dtype=torch.float32)}

    save_file(t1, str(dir1 / "adapter_model.safetensors"))
    save_file(t2, str(dir2 / "adapter_model.safetensors"))

    out_dir = tmp_path / "soup"
    out_file = average_safetensors_weights([dir1, dir2], out_dir)

    assert out_file.exists()
    soup_tensors = load_file(str(out_file))
    # (2+4)/2 = 3.0, (4+8)/2 = 6.0
    expected = torch.tensor([3.0, 6.0], dtype=torch.float32)
    assert torch.allclose(soup_tensors["weight"], expected)
