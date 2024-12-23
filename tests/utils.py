import os
import timm
import torch
import unittest

from packaging.version import Version


has_timm_test_models = Version(timm.__version__) >= Version("1.0.12")
default_device = "cuda" if torch.cuda.is_available() else "cpu"


def get_commit_message():
    commit_msg = os.getenv("COMMIT_MESSAGE", "")
    return commit_msg.lower()


# Check both environment variables and commit message
commit_message = get_commit_message()
RUN_ALL_ENCODERS = (
    os.getenv("RUN_ALL_ENCODERS", "false").lower() in ["true", "1", "y", "yes"]
    or "run-all-encoders" in commit_message
)

RUN_SLOW = (
    os.getenv("RUN_SLOW", "false").lower() in ["true", "1", "y", "yes"]
    or "run-slow" in commit_message
)


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
