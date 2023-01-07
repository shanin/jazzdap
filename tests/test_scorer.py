import sys

sys.path.append("src")


def test_always_passes():
    assert True


def test_import():
    from model import CRNN
    from utils import load_config
    from solo import construct_solo_class
