# -*- coding: utf-8 -*-
"""
core_errors.py
Общие исключения для системы.
"""


class BotError(Exception):
    """ Базовая ошибка системы. """


class ConfigError(BotError):
    """ Ошибка конфигурации/валидации. """


class DataError(BotError):
    """ Ошибка данных/источника. """


class ExecutionError(BotError):
    """ Ошибка исполнения ордеров. """


class RiskViolation(BotError):
    """ Нарушение риск-ограничений. """


class QuantizeError(BotError):
    """ Ошибка квантизации цен/количеств. """


class BacktestError(BotError):
    """ Ошибка работы бэктест-движка. """
