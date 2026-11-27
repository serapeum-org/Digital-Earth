"""Package-level options a caller sets to steer digitalearth without reaching into a backend (#573)."""

import pytest

import digitalearth


@pytest.fixture(autouse=True)
def _restore_options():
    """Reset the option store after each test so one test's writes do not leak into another."""
    from digitalearth.base.options import _reset_options

    yield
    _reset_options()


def test_three_d_off_screen_defaults_to_auto():
    """The out-of-the-box value of the 3-D off-screen option is ``"auto"`` (#573)."""
    assert digitalearth.get_option("three_d.off_screen") == "auto"


@pytest.mark.parametrize("value", [True, False, "auto"])
def test_set_then_get_round_trips_each_valid_value(value):
    """Every accepted value round-trips through ``set_option``/``get_option`` (#573)."""
    digitalearth.set_option("three_d.off_screen", value)
    assert digitalearth.get_option("three_d.off_screen") == value


def test_set_option_rejects_an_unknown_name():
    """Setting an option digitalearth does not define is refused, echoing the rejected name (#573)."""
    with pytest.raises(ValueError, match="no_such_option"):
        digitalearth.set_option("three_d.no_such_option", True)


def test_get_option_rejects_an_unknown_name():
    """Reading an undefined option is refused rather than returning a silent ``None`` (#573)."""
    with pytest.raises(ValueError, match="no_such_option"):
        digitalearth.get_option("three_d.no_such_option")


def test_set_option_rejects_an_invalid_value():
    """A value outside ``True``/``False``/``"auto"`` for the off-screen option is refused (#573)."""
    with pytest.raises(ValueError, match="off_screen"):
        digitalearth.set_option("three_d.off_screen", "sometimes")
