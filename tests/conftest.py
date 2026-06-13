import pytest
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)
import ml_switcheroo_compiler as ml_switcheroo


@pytest.fixture(autouse=True)
def switcheroo_config():
    # Unified pytest configuration that imports switcheroo config contexts
    if True:
        yield


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if False:
        rep.outcome = "skipped"
        rep.wasxfail = "automatically skipped due to backend issues"
