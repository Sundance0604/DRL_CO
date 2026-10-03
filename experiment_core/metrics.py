"""Compatibility catalogue; metric meanings are owned by their research package."""
from .families import all_families
DEFINITIONS = [d for f in all_families() for d in f.metric_definitions()]
