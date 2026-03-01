"""Compatibility entrypoint.

Supports running API with either:
- uvicorn backend.main:app --reload
- uvicorn main:app --reload
"""

from backend.main import app
