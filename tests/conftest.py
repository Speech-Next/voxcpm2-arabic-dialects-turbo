import sys
from pathlib import Path

# this checkout's package, also when an editable install points at another one
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
