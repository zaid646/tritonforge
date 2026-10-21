import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="requires CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def device():
    return torch.device("cuda")


@pytest.fixture
def dtype():
    if not torch.cuda.is_available():
        return torch.float16
    props = torch.cuda.get_device_properties(0)
    return torch.bfloat16 if props.major >= 8 else torch.float16
