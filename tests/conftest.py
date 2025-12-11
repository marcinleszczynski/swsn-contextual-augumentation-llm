"""
Pytest configuration and fixtures.
"""

import sys
import os

# Add project root to path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
