"""Pytest adapter: script test arguments would otherwise be collected as fixtures."""

from __future__ import annotations

import test_kernel


def test_kernel_suite() -> None:
    assert test_kernel.main() == 0, "Kernel checks failed"


if __name__ == "__main__":
    test_kernel_suite()
    print("All checks passed")
