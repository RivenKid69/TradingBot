"""Thin compatibility wrapper for ServiceSignalRunner.

This module previously contained the business logic for running
realtime signalers. The logic has been moved to
:mod:`service_signal_runner`. Importing :class:`SignalRunner` from this
module now simply re-exports :class:`ServiceSignalRunner`.
"""

from service_signal_runner import ServiceSignalRunner as SignalRunner

__all__ = ["SignalRunner", "ServiceSignalRunner"]
