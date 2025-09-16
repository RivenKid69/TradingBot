"""Stub components used for dependency injection tests."""

from __future__ import annotations


class DummyMarketData:
    def __init__(self, *_, **__):
        self.started = True


class DummyFeaturePipe:
    def __init__(self, *_, **__):
        self.ready = True


class DummyPolicy:
    def __init__(self, *_, **__):
        self.initialized = True


class DummyRiskGuards:
    def __init__(self, *_, **__):
        self.enabled = True


class DummyExecutor:
    def __init__(self, *_, quantizer=None, **__):
        self.quantizer = quantizer
        self.attached = False

    def attach(self):
        self.attached = True
