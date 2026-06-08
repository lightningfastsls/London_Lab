"""Exclude archived dead-collection tests from pytest discovery.

These test modules error at collection because the code they target was
removed or archived (see README.md in this directory). They are preserved
here as spec/provenance, NOT run. `collect_ignore_glob` stops pytest from
importing them so `pytest tests/` collects cleanly.

To revive one: restore its target module/script, move the test back up to
tests/, and confirm it collects.
"""

collect_ignore_glob = ["test_*.py"]
