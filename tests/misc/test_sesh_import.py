from __future__ import annotations


def test_sesh_import() -> None:
    """
    A baseline sanity test to verify that the sesh package imports correctly.
    """
    import sesh  # noqa: F401

    assert True
