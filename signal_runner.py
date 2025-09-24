"""Thin compatibility wrapper for :class:`service_signal_runner.ServiceSignalRunner`.

Historically the trading services imported ``SignalRunner`` from this module.
After the refactor in which the implementation moved to
``service_signal_runner.ServiceSignalRunner`` the intent was to keep this
module as a compatibility shim that re-exports the new class.  The previous
implementation only defined the alias ``SignalRunner`` and forgot to export
``ServiceSignalRunner`` itself, meaning ``from signal_runner import
ServiceSignalRunner`` raised :class:`ImportError`.  Some callers – including
tests – still import the class directly under its original name, so we need to
expose both symbols.
"""

from service_signal_runner import ServiceSignalRunner

# Re-export both the legacy name (SignalRunner) and the new explicit class
# name to preserve backwards compatibility.
SignalRunner = ServiceSignalRunner

__all__ = ["SignalRunner", "ServiceSignalRunner"]
