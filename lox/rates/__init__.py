from __future__ import annotations

"""
Rates / yield curve regime.

Goal:
- Provide a clean, ML-friendly summary of the rates backdrop (level, slope, momentum).
- Keep this separate from `funding` (secured funding plumbing) to avoid conceptual overlap.
"""


