"""evalconfidence: decision-grade statistics on top of existing eval frameworks."""

from .adapters import from_dataframe, from_inspect
from .compare import ComparisonResult, compare
from .power import PowerResult, power
from .stderr import standard_error
from .types import ItemResult, SEResult

__version__ = "0.1.0.dev0"

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
