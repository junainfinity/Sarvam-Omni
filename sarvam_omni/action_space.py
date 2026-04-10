"""Agentic action space definitions and parsing for GUI interaction."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Action:
    action_type: str
    x: Optional[float] = None
    y: Optional[float] = None
    text: Optional[str] = None
    direction: Optional[str] = None
    start_x: Optional[float] = None
    start_y: Optional[float] = None
    end_x: Optional[float] = None
    end_y: Optional[float] = None


# Canonical action vocabulary (Smol2Operator-style, normalized [0,1] coordinates)
ACTION_TYPES = ["click", "type", "scroll", "drag", "long_press", "done"]

ACTION_FORMAT_EXAMPLES = """click(x=0.45, y=0.72)
type(text="search query")
scroll(direction="down")
drag(startX=0.1, startY=0.2, endX=0.5, endY=0.8)
long_press(x=0.3, y=0.5)
done()"""

# Regex patterns for parsing model output
_CLICK_RE = re.compile(r'click\(x=([\d.]+),\s*y=([\d.]+)\)')
_TYPE_RE = re.compile(r'type\(text="([^"]+)"\)')
_SCROLL_RE = re.compile(r'scroll\(direction="(\w+)"\)')
_DRAG_RE = re.compile(
    r'drag\(startX=([\d.]+),\s*startY=([\d.]+),\s*endX=([\d.]+),\s*endY=([\d.]+)\)'
)
_LONG_PRESS_RE = re.compile(r'long_press\(x=([\d.]+),\s*y=([\d.]+)\)')
_DONE_RE = re.compile(r'done\(\)')


def parse_action(text: str) -> Optional[Action]:
    """Parse a single action string into an Action object. Returns None if unparseable."""
    text = text.strip()

    m = _CLICK_RE.search(text)
    if m:
        return Action("click", x=float(m.group(1)), y=float(m.group(2)))

    m = _TYPE_RE.search(text)
    if m:
        return Action("type", text=m.group(1))

    m = _SCROLL_RE.search(text)
    if m:
        return Action("scroll", direction=m.group(1))

    m = _DRAG_RE.search(text)
    if m:
        return Action(
            "drag",
            start_x=float(m.group(1)), start_y=float(m.group(2)),
            end_x=float(m.group(3)), end_y=float(m.group(4)),
        )

    m = _LONG_PRESS_RE.search(text)
    if m:
        return Action("long_press", x=float(m.group(1)), y=float(m.group(2)))

    if _DONE_RE.search(text):
        return Action("done")

    return None


def parse_actions(text: str) -> list[Action]:
    """Parse all actions from model output text."""
    actions = []
    for line in text.strip().split("\n"):
        action = parse_action(line)
        if action:
            actions.append(action)
    return actions
