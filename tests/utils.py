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
