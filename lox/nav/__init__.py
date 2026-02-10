"""NAV (net asset value) tracking utilities.

This module provides lightweight, spreadsheet-friendly tracking of:
- cash flows (deposits/withdrawals)
- NAV snapshots (account equity at a point in time)

The goal is to compute leak-proof-ish *account-level* performance series that
correctly adjusts for user contributions/withdrawals.
"""

