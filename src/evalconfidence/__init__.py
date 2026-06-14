"""evalconfidence: decision-grade statistics on top of existing eval frameworks."""

from importlib.metadata import PackageNotFoundError, version

from .adapters import from_dataframe, from_inspect
from .compare import ComparisonResult, compare
from .power import PowerResult, power
from .stderr import standard_error
from .types import ItemResult, SEResult

try:
    __version__ = version("evalconfidence")
except PackageNotFoundError:  # running from source without an install
    __version__ = "0.0.0+unknown"

__all__ = [
    "ComparisonResult",
    "ItemResult",
    "PowerResult",
    "SEResult",
    "compare",
    "from_dataframe",
    "from_inspect",
    "power",
    "standard_error",
    "__version__",
]
