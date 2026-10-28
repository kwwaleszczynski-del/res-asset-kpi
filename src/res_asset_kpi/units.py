"""Units of measurement: a small registry and conversion between units of the same dimension.

Every unit belongs to one dimension and is stored as ``(dimension, scale, offset)`` against the canonical unit
of that dimension, so that ``canonical = value * scale + offset``. Unit names are case-insensitive and
tolerate the usual spellings (``degC``, ``°C``, ``W/m²``, ``W/m2``).

| dimension | canonical unit | other units |
|---|---|---|
| power | kW | W, MW, GW |
| energy | kWh | Wh, MWh, GWh, J, MJ, GJ |
| irradiance | W/m2 | kW/m2 |
| irradiation | kWh/m2 | Wh/m2, MJ/m2 |
| temperature | degC | degF, K |
| speed | m/s | km/h, mph, kn |
| rotation | rpm | rad/s |
| fraction | fraction | %, 1, pu |
| angle | deg | rad |
| pressure | Pa | hPa, kPa, mbar, bar |
| density | kg/m3 |  |
| length | mm | cm, m |
| time | h | s, min, d |
"""

from __future__ import annotations

import math
from typing import TypeVar

import numpy as np
import pandas as pd

__all__ = ["UNITS", "canonical_unit", "convert", "dimension", "normalise_unit"]

T = TypeVar("T", float, np.ndarray, pd.Series, pd.DataFrame)

# unit -> (dimension, scale, offset) such that canonical = value * scale + offset
UNITS: dict[str, tuple[str, float, float]] = {
    "kw": ("power", 1.0, 0.0),
    "w": ("power", 1e-3, 0.0),
    "mw": ("power", 1e3, 0.0),
    "gw": ("power", 1e6, 0.0),
    "kwh": ("energy", 1.0, 0.0),
    "wh": ("energy", 1e-3, 0.0),
    "mwh": ("energy", 1e3, 0.0),
    "gwh": ("energy", 1e6, 0.0),
    "j": ("energy", 1.0 / 3.6e6, 0.0),
    "mj": ("energy", 1.0 / 3.6, 0.0),
    "gj": ("energy", 1e3 / 3.6, 0.0),
    "w/m2": ("irradiance", 1.0, 0.0),
    "kw/m2": ("irradiance", 1e3, 0.0),
    "kwh/m2": ("irradiation", 1.0, 0.0),
    "wh/m2": ("irradiation", 1e-3, 0.0),
    "mj/m2": ("irradiation", 1.0 / 3.6, 0.0),
    "degc": ("temperature", 1.0, 0.0),
    "c": ("temperature", 1.0, 0.0),
    "degf": ("temperature", 5.0 / 9.0, -32.0 * 5.0 / 9.0),
    "f": ("temperature", 5.0 / 9.0, -32.0 * 5.0 / 9.0),
    "k": ("temperature", 1.0, -273.15),
    "m/s": ("speed", 1.0, 0.0),
    "km/h": ("speed", 1.0 / 3.6, 0.0),
    "mph": ("speed", 0.44704, 0.0),
    "kn": ("speed", 1852.0 / 3600.0, 0.0),
    "rpm": ("rotation", 1.0, 0.0),
    "rad/s": ("rotation", 60.0 / (2.0 * math.pi), 0.0),
    "fraction": ("fraction", 1.0, 0.0),
    "1": ("fraction", 1.0, 0.0),
    "pu": ("fraction", 1.0, 0.0),
    "%": ("fraction", 0.01, 0.0),
    "deg": ("angle", 1.0, 0.0),
    "rad": ("angle", 180.0 / math.pi, 0.0),
    "pa": ("pressure", 1.0, 0.0),
    "hpa": ("pressure", 100.0, 0.0),
    "mbar": ("pressure", 100.0, 0.0),
    "kpa": ("pressure", 1e3, 0.0),
    "bar": ("pressure", 1e5, 0.0),
    "kg/m3": ("density", 1.0, 0.0),
    "mm": ("length", 1.0, 0.0),
    "cm": ("length", 10.0, 0.0),
    "m": ("length", 1e3, 0.0),
    "h": ("time", 1.0, 0.0),
    "s": ("time", 1.0 / 3600.0, 0.0),
    "min": ("time", 1.0 / 60.0, 0.0),
    "d": ("time", 24.0, 0.0),
}
"""Unit registry: name -> (dimension, scale, offset), ``canonical = value * scale + offset``."""
_CANONICAL = {dim: unit for unit, (dim, scale, offset) in UNITS.items() if scale == 1.0 and offset == 0.0}
_CANONICAL.update({"temperature": "degc", "fraction": "fraction"})
_ALIASES = {"°": "deg", "²": "2", "³": "3", " ": "", "^": "", "celsius": "c", "kelvin": "k", "percent": "%"}


def normalise_unit(unit: str) -> str:
    """Registry key of a unit name.

    Parameters
    ----------
    unit : str
        Unit as written in a file header or a mapping, e.g. ``"°C"``, ``"W/m²"``, ``"MWh"``.

    Returns
    -------
    str
        The lower-case key used in [`UNITS`][res_asset_kpi.units.UNITS].

    Raises
    ------
    ValueError
        If the unit is not in the registry.

    Examples
    --------
    >>> normalise_unit("W/m²")
    'w/m2'
    >>> normalise_unit("°C")
    'degc'
    """
    key = str(unit).strip().lower()
    for old, new in _ALIASES.items():
        key = key.replace(old, new)
    if key == "degk":
        key = "k"
    if key not in UNITS:
        raise ValueError(f"unknown unit {unit!r}; known units: {sorted(UNITS)}")
    return key


def dimension(unit: str) -> str:
    """Dimension of a unit (``"power"``, ``"energy"``, ...).

    Examples
    --------
    >>> dimension("MWh")
    'energy'
    """
    return UNITS[normalise_unit(unit)][0]


def canonical_unit(unit: str) -> str:
    """Canonical unit of the dimension ``unit`` belongs to.

    Examples
    --------
    >>> canonical_unit("MW")
    'kw'
    >>> canonical_unit("degF")
    'degc'
    """
    return _CANONICAL[dimension(unit)]


def convert(values: T, from_unit: str, to_unit: str) -> T:
    """Convert values between two units of the same dimension.

    Parameters
    ----------
    values : float, numpy.ndarray, pandas.Series or pandas.DataFrame
        Values in ``from_unit``; NaN stays NaN.
    from_unit, to_unit : str
        Unit names (see the table in the module description).

    Returns
    -------
    Same type as ``values``
        Values in ``to_unit``.

    Raises
    ------
    ValueError
        If a unit is unknown or the two units measure different dimensions.

    Examples
    --------
    >>> convert(1500.0, "W", "kW")
    1.5
    >>> round(convert(32.0, "degF", "degC"), 6)
    0.0
    >>> convert(0.5, "fraction", "%")
    50.0
    """
    d_from, s_from, o_from = UNITS[normalise_unit(from_unit)]
    d_to, s_to, o_to = UNITS[normalise_unit(to_unit)]
    if d_from != d_to:
        raise ValueError(f"cannot convert {from_unit!r} ({d_from}) to {to_unit!r} ({d_to})")
    canonical = values * s_from + o_from
    return (canonical - o_to) / s_to
