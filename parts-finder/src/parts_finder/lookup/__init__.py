"""Category-specific lookup modules.

Each module implements a cascade strategy that progressively relaxes
matching criteria until a result is found or all tiers are exhausted.
"""

from parts_finder.lookup.brakes import BrakeLookup
from parts_finder.lookup.bulbs import BulbLookup
from parts_finder.lookup.coolant import CoolantLookup
from parts_finder.lookup.filters import FilterLookup

__all__ = ["BrakeLookup", "BulbLookup", "CoolantLookup", "FilterLookup"]