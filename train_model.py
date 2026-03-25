"""Backward-compatible entrypoint for training.

This script intentionally reuses the project pipeline in main.py so preprocessing,
training, and evaluation stay in one canonical flow.
"""

from main import main


if __name__ == "__main__":
    main()
