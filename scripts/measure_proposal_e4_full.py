"""Use the frozen, corrected adapter package for full proposal E4."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
import tf_ovos
assert Path(tf_ovos.__file__).resolve().is_relative_to(Path(__file__).resolve().parent / 'src')
import measure_e4_full
if __name__ == '__main__':
    measure_e4_full.main()
