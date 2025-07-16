import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pricepulsebot.handlers import milestone_step, milestones_crossed  # noqa: E402


def test_milestone_step():
    assert milestone_step(2000) == 100.0
    assert milestone_step(500) == 10.0
    assert milestone_step(50) == 1.0
    assert milestone_step(5) == 0.1
    assert milestone_step(0.5) == 0.01
    assert milestone_step(0.05) == 0.001
    assert milestone_step(0.005) == 0.0001
    assert milestone_step(0.0005) == 0.00001
    assert milestone_step(0.00005) == 0.000001


def test_milestones_crossed_up():
    levels = milestones_crossed(95, 105)
    assert levels == [100]


def test_milestones_crossed_down():
    levels = milestones_crossed(160.1, 159.8)
    assert levels == [160]
