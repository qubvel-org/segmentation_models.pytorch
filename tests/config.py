import os


def get_commit_message():
    # Get commit message from CI environment variables
    # Common CI systems store commit messages in different env vars
    commit_msg = os.getenv("COMMIT_MESSAGE", "")  # Generic
    if not commit_msg:
        commit_msg = os.getenv("CI_COMMIT_MESSAGE", "")  # GitLab CI
    if not commit_msg:
        commit_msg = os.getenv("GITHUB_EVENT_HEAD_COMMIT_MESSAGE", "")  # GitHub Actions
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
