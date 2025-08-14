# make sure cli loads without error
# this code is shamelessly copied from asapdiscovery-cli/asapdiscovery/cli/tests/test_meta_cli.py

from harbor.cli import cli as harbor_cli
from click.testing import CliRunner
import traceback


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


def test_toplevel_runnable():
    runner = CliRunner()
    args = ["--help"]
    result = runner.invoke(harbor_cli, args)
    assert click_success(result)
