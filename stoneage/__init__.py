"""
stoneage: Python port of Greg Balco's v3 cosmogenic nuclide exposure age calculator.

Original MATLAB code: https://stoneage.ice-d.org/math/v3/
Original author: Greg Balco, Berkeley Geochronology Center (balcs@bgc.org)
"""

from .constants import make_consts
from .workflow import ProductionRateWorkflow
from .atmosphere import ERA40atm, antatm
from .scaling import stone2000, get_LmSF, get_LSDnSF
from .corrections import thickness, Lsp
from .cutoff_rigidity import get_DipRc
from .calculator import get_ages
from .input_parser import parse_v3_input, csv_to_v3
from .summarize import summarize_ages

__version__ = "3.0.2"
__all__ = [
    "make_consts",
    "ERA40atm",
    "antatm",
    "stone2000",
    "get_LmSF",
    "get_LSDnSF",
    "thickness",
    "Lsp",
    "get_DipRc",
    "get_ages",
    "parse_v3_input",
    "csv_to_v3",
    "summarize_ages",
]
