import os
import re
import timm
import torch
import unittest

from git import Repo
from typing import List
from packaging.version import Version


has_timm_test_models = Version(timm.__version__) >= Version("1.0.12")
default_device = "cuda" if torch.cuda.is_available() else "cpu"

YES_LIST = ["true", "1", "y", "yes"]
RUN_ALL_ENCODERS = os.getenv("RUN_ALL_ENCODERS", "false").lower() in YES_LIST
RUN_SLOW = os.getenv("RUN_SLOW", "false").lower() in YES_LIST
RUN_ALL = os.getenv("RUN_ALL", "false").lower() in YES_LIST


def slow_test(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.

    """
    return unittest.skipUnless(RUN_SLOW, "test is slow")(test_case)


def requires_timm_greater_or_equal(version: str):
    timm_version = Version(timm.__version__)
    provided_version = Version(version)
    return unittest.skipUnless(
        timm_version >= provided_version,
        f"timm version {timm_version} is less than {provided_version}",
    )


def requires_torch_greater_or_equal(version: str):
    torch_version = Version(torch.__version__)
    provided_version = Version(version)
    return unittest.skipUnless(
        torch_version >= provided_version,
        f"torch version {torch_version} is less than {provided_version}",
    )


def check_run_test_on_diff_or_main(filepath_patterns: List[str]):
    if RUN_ALL:
        return True

    try:
        repo = Repo(".")
        current_branch = repo.active_branch.name
        diff_files = repo.git.diff("main", name_only=True).splitlines()

    except Exception:
        return True

    if current_branch == "main":
        return True

    for pattern in filepath_patterns:
        for file_path in diff_files:
            if re.search(pattern, file_path):
                return True

    return False


def check_two_models_strictly_equal(
    model_a: torch.nn.Module, model_b: torch.nn.Module, input_data: torch.Tensor
) -> None:
    for (k1, v1), (k2, v2) in zip(
        model_a.state_dict().items(), model_b.state_dict().items()
    ):
        assert k1 == k2, f"Key mismatch: {k1} != {k2}"
        torch.testing.assert_close(
            v1, v2, msg=f"Tensor mismatch at key '{k1}':\n{v1} !=\n{v2}"
        )

    model_a.eval()
    model_b.eval()
    with torch.inference_mode():
        output_a = model_a(input_data)
        output_b = model_b(input_data)

    torch.testing.assert_close(output_a, output_b)
