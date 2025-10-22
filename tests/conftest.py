import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path so `import phase1` works when running pytest from anywhere
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

