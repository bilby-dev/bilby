"""Run the :mod:`doctest` examples embedded in bilby docstrings.

This test file wires bilby's docstring examples into the standard
``unittest`` test suite via :class:`doctest.DocTestSuite`.  Modules are
added to the suite explicitly rather than via a recursive walk because
some bilby submodules have heavy optional dependencies (``lalsuite``,
``gwpy``, ...) that are not guaranteed to be importable in every test
environment.  When adding a new doctest to bilby, append the parent
module to :data:`MODULES_WITH_DOCTESTS` below.

The test is discovered by :mod:`pytest` in the usual way, so it runs as
part of the standard ``pytest test`` invocation, and is also runnable
directly via ``python -m unittest test.core.doctest_test``.
"""
import doctest
import importlib
import unittest


# Add the importable dotted name of any bilby module that contains
# doctest-format ``>>>`` examples.  Only list modules that are safe to
# import in a minimal test environment (no LAL, no gwpy, no samplers).
MODULES_WITH_DOCTESTS = [
    "bilby.core.utils.introspection",
]


def load_tests(loader, tests, ignore):
    """unittest ``load_tests`` protocol hook.

    Collects doctests from every module listed in
    :data:`MODULES_WITH_DOCTESTS` and appends them to the existing
    ``TestSuite``.  Modules that fail to import (for example because an
    optional dependency is missing) are silently skipped so that the
    bulk of the suite still runs.
    """
    for module_name in MODULES_WITH_DOCTESTS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        try:
            suite = doctest.DocTestSuite(
                module,
                optionflags=(
                    doctest.ELLIPSIS
                    | doctest.NORMALIZE_WHITESPACE
                    | doctest.IGNORE_EXCEPTION_DETAIL
                ),
            )
        except ValueError:
            # DocTestSuite raises ValueError if the module contains no
            # doctests at all.  This is not a failure for us — modules
            # are expected to grow doctests over time.
            continue
        tests.addTests(suite)
    return tests


class TestDoctestInfrastructure(unittest.TestCase):
    """Sanity check that the doctest wiring itself is functional."""

    def test_modules_with_doctests_is_populated(self):
        self.assertTrue(
            len(MODULES_WITH_DOCTESTS) > 0,
            "MODULES_WITH_DOCTESTS should list at least one module so "
            "that the doctest runner has something to execute.",
        )

    def test_listed_modules_are_importable(self):
        for module_name in MODULES_WITH_DOCTESTS:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_introspection_module_has_at_least_one_doctest(self):
        from bilby.core.utils import introspection

        finder = doctest.DocTestFinder()
        found = finder.find(introspection)
        total_examples = sum(len(t.examples) for t in found)
        self.assertGreater(
            total_examples,
            0,
            "bilby.core.utils.introspection should contain at least "
            "one doctest example.",
        )


if __name__ == "__main__":
    unittest.main()
