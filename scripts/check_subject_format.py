"""Validate Bilby PR titles and commit subjects using its SciPy-based convention."""

import argparse
import json
from pathlib import Path
import re
import sys


ACRONYMS = "API|BENCH|BLD|BUG|DEP|DEV|DOC|ENH|MAINT|REV|STY|TST|REL"
PATTERN = rf"(?:{ACRONYMS})(?:/(?:{ACRONYMS}))*(?:\([^\s()]+\))?: [^\s\r\n][^\r\n]*"


def main():
    """Read a PR event or commit message file and validate its subject."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--github-event", type=Path, help="GitHub PR event JSON file")
    source.add_argument("--commit-msg-file", type=Path, help="Git commit message file")
    args = parser.parse_args()

    if args.github_event:
        title = json.loads(args.github_event.read_text(encoding="utf-8"))["pull_request"]["title"]
    else:
        message = args.commit_msg_file.read_text(encoding="utf-8")
        title = message.partition("\n")[0]

    if re.fullmatch(PATTERN, title) is not None:
        return 0

    prefix = "::error::" if args.github_event else ""
    print(
        prefix + "Use the Bilby subject format 'ACRONYM: Summary message' "
        "(e.g. 'BUG: Fix prior sampling'). Allowed acronyms: "
        + ACRONYMS.replace("|", ", ")
        + ". Multiple acronyms may be joined with '/', e.g. 'MAINT/TST: Update tests'. "
        "An optional scope may follow the acronym(s), e.g. 'BUG(gw): Fix sampling'. "
        "Scopes must be nonempty and contain no whitespace or parentheses. "
        "See CONTRIBUTING.md#pull-requests.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
