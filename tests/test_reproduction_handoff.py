"""Relocation must not silently fall back to old paths or alter table semantics."""
from pathlib import Path
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from reproduce_e4 import relocate
from reproduce_published_statistics import compare


def test_relocation_longest_component_prefix(tmp_path):
    mapping = {'/old': str(tmp_path / 'general'), '/old/data': str(tmp_path / 'dataset')}
    assert relocate('/old/data/a.png', mapping) == tmp_path / 'dataset/a.png'
    assert relocate('/old/database/a.png', mapping) == tmp_path / 'general/database/a.png'


def test_relocation_refuses_hidden_old_path(tmp_path):
    with pytest.raises(ValueError, match='No path mapping'):
        relocate('/old-other/a.png', {'/old': str(tmp_path)})


def test_statistics_comparison_refuses_missing_row():
    with pytest.raises(ValueError):
        compare([dict(method='a', value=1)], [], ['method'], ['value'])


def test_statistics_comparison_refuses_changed_number():
    with pytest.raises(ValueError, match='mismatch'):
        compare([dict(method='a', value=1.0001)], [dict(method='a', value='1')], ['method'], ['value'])
