from pricepulsebot.handlers import (format_price, milestone_step,
                                    milestones_crossed)


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


def test_format_price_trailing_zero():
    assert format_price(0.6) == "0.60"


def test_format_price_small_value():
    assert format_price(3.7e-05) == "0.000037"


def test_milestone_message_format():
    level = 0.6
    price = 0.65
    msg = f"BTC breaks through ${format_price(level)} " f"(now ${format_price(price)})"
    assert msg == "BTC breaks through $0.60 (now $0.65)"
