"""
Syntax highlighting matching the code blocks on https://bilby-dev.github.io.

The same dark style is used for both the light and dark themes, as on the
website.
"""

from pygments.style import Style
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
)

BACKGROUND = "#0e2238"
FOREGROUND = "#e4eef8"
MUTED = "#8aa6c1"
KEYWORD = "#ff9fb5"
STRING = "#a8e0a0"
COMMENT = "#7f97ae"
NUMBER = "#ffcf85"
FUNCTION = "#8fd0ff"
BUILTIN = "#c7a6ff"


class BilbyStyle(Style):
    name = "bilby"
    background_color = BACKGROUND
    highlight_color = "#1b3a5c"
    line_number_color = MUTED
    line_number_background_color = BACKGROUND

    styles = {
        Text: FOREGROUND,
        Error: "#ff7b72",
        Comment: f"italic {COMMENT}",
        Comment.Preproc: f"noitalic {COMMENT}",
        Keyword: KEYWORD,
        Keyword.Constant: BUILTIN,
        Operator: FOREGROUND,
        Operator.Word: KEYWORD,
        Punctuation: FOREGROUND,
        Name: FOREGROUND,
        Name.Builtin: BUILTIN,
        Name.Builtin.Pseudo: BUILTIN,
        Name.Function: FUNCTION,
        Name.Function.Magic: FUNCTION,
        Name.Class: FUNCTION,
        Name.Decorator: NUMBER,
        Name.Exception: BUILTIN,
        Name.Namespace: FOREGROUND,
        Name.Tag: KEYWORD,
        Name.Attribute: FUNCTION,
        String: STRING,
        String.Doc: f"italic {COMMENT}",
        String.Escape: NUMBER,
        String.Interpol: NUMBER,
        Number: NUMBER,
        Generic.Prompt: MUTED,
        Generic.Output: MUTED,
        Generic.Traceback: "#ff7b72",
        Generic.Error: "#ff7b72",
        Generic.Heading: f"bold {FUNCTION}",
        Generic.Subheading: FUNCTION,
        Generic.Deleted: "#ff7b72",
        Generic.Inserted: STRING,
        Generic.Emph: "italic",
        Generic.Strong: "bold",
    }
