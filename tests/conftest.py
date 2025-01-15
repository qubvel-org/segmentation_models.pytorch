def pytest_addoption(parser):
    parser.addoption(
        "--non-marked-only", action="store_true", help="Run only non-marked tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--non-marked-only"):
        non_marked_items = []
        for item in items:
            # Check if the test has no marks
            if not item.own_markers:
                non_marked_items.append(item)

        # Update the test collection to only include non-marked tests
        items[:] = non_marked_items
