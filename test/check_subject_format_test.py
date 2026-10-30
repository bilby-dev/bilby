"""Exercise the shared title validator through both command-line interfaces."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_subject_format.py"


class TestSubjectFormat(unittest.TestCase):
    def run_validator(self, title, github_event):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "message"
            if github_event:
                source.write_text(json.dumps({"pull_request": {"title": title}}), encoding="utf-8")
                option = "--github-event"
            else:
                source.write_text(title, encoding="utf-8")
                option = "--commit-msg-file"
            return subprocess.run(
                [sys.executable, str(SCRIPT), option, str(source)],
                capture_output=True, text=True,
            )

    def test_valid_titles(self):
        titles = [
            f"{acronym}: Valid summary"
            for acronym in "API BENCH BLD BUG DEP DEV DOC ENH MAINT REV STY TST REL".split()
        ] + [
            'MAINT/TST: Update tests', 'DOC: Explain `café` and $(echo example)',
            'BUG(gw): Fix sampling', 'MAINT/TST(core.prior): Update tests',
            'DOC(gw/likelihood): Update documentation', 'DEV(pre-commit): Update hooks',
        ]
        for title in titles:
            for github_event in (True, False):
                with self.subTest(title=title, github_event=github_event):
                    result = self.run_validator(title, github_event)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_titles(self):
        for title in ('', 'Fix sampling', 'bug: Fix sampling', 'FEAT: Add sampling',
                      'BUG:Fix sampling', 'BUG: ', 'BUG:   ', ' BUG: Fix sampling',
                      'BUG/INVALID: Fix sampling', 'BUG/: Fix sampling',
                      'BUG(): Fix sampling', 'BUG( ): Fix sampling',
                      'BUG(core prior): Fix sampling', 'BUG(core\tprior): Fix sampling',
                      'BUG((gw)): Fix sampling', 'BUG(gw: Fix sampling',
                      'BUG(gw)): Fix sampling', 'BUG (gw): Fix sampling',
                      'BUG(gw): ', 'BUG(gw)/TST: Fix sampling'):
            for github_event in (True, False):
                with self.subTest(title=title, github_event=github_event):
                    result = self.run_validator(title, github_event)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn("Bilby subject format", result.stderr)
                    self.assertEqual(result.stderr.startswith("::error::"), github_event)

    def test_commit_body_is_allowed(self):
        result = self.run_validator('BUG: Fix sampling\n\nExplain the fix.\n', False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_multiline_pr_title_is_rejected(self):
        result = self.run_validator('BUG: Fix sampling\nExtra line', True)
        self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()
