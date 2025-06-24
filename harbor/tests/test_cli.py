# make sure cli loads without error

from harbor.cli import cli as harbor_cli


def test_cli():
    """Test that the CLI loads without error."""
    try:
        harbor_cli()
    except Exception as e:
        assert False, f"CLI failed to load: {e}"
    assert True, "CLI loaded successfully"
