# ## Имя файла: data_validation.py
import pandas as pd
import numpy as np
import re

class DataValidator:
    """
    Класс для строгой, атомарной верификации данных временных рядов (OHLCV).
    Гарантирует целостность, консистентность и непрерывность данных перед
    их использованием в ML-системах.
    """

    def validate(self, df: pd.DataFrame, frequency: str = None):
        """
        Основной публичный метод для запуска всех проверок в строгой последовательности.
        Процесс прерывается немедленно при обнаружении первой ошибки.

        Args:
            df (pd.DataFrame): Входной датафрейм для проверки.
                               Ожидается, что у него будет DatetimeIndex.
            frequency (str, optional): Ожидаемая частота временного ряда (например, '1T', '1H', '1D').
                                       Если None, будет предпринята попытка автоопределения.

        Raises:
            ValueError: Если какая-либо из проверок не пройдена.
        """
        print("Запуск процесса валидации данных...")
        self._check_for_nulls(df)
        self._check_values_are_positive(df)
        self._check_ohlc_invariants(df)
        self._check_timestamp_continuity(df, frequency)
        self._check_schema_and_order(df)
        self._check_no_pii(df)
        print("Валидация данных успешно завершена.")
        return df

    def _check_for_nulls(self, df: pd.DataFrame):
        """Проверяет наличие NaN или inf значений в ключевых колонках."""
        key_columns = ['open', 'high', 'low', 'close', 'quote_asset_volume']
        # Проверяем только те колонки, которые существуют в df
        cols_to_check = [col for col in key_columns if col in df.columns]

        if df[cols_to_check].isnull().values.any():
            nan_info = df[cols_to_check].isnull().sum()
            nan_info = nan_info[nan_info > 0]
            raise ValueError(f"Обнаружены NaN значения в данных:\n{nan_info}")

        if np.isinf(df[cols_to_check]).values.any():
            inf_info = np.isinf(df[cols_to_check]).sum()
            inf_info = inf_info[inf_info > 0]
            raise ValueError(f"Обнаружены бесконечные (inf) значения в данных:\n{inf_info}")

    def _check_values_are_positive(self, df: pd.DataFrame):
        """Убеждается, что значения в колонках цен и объёмов строго больше нуля."""
        positive_columns = ['open', 'high', 'low', 'close', 'quote_asset_volume']
        # Проверяем только те колонки, которые существуют в df
        cols_to_check = [col for col in positive_columns if col in df.columns]

        # Используем .le() для сравнения <= 0 и находим проблемные места
        violations = df[cols_to_check].le(0)
        if violations.any().any():
            first_violation_idx = violations.any(axis=1).idxmax()
            violation_details = df.loc[first_violation_idx, cols_to_check]
            raise ValueError(
                f"Обнаружены нулевые или отрицательные значения. "
                f"Первое нарушение в индексе {first_violation_idx}:\n{violation_details}"
            )

    def _check_ohlc_invariants(self, df: pd.DataFrame):
        """Проверяет инварианты OHLC: high - максимальное значение, low - минимальное."""
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные OHLC-колонки: {', '.join(missing_columns)}")
        checks = {
            "high >= low": (df['high'] < df['low']),
            "high >= open": (df['high'] < df['open']),
            "high >= close": (df['high'] < df['close']),
            "low <= open": (df['low'] > df['open']),
            "low <= close": (df['low'] > df['close']),
        }

        for description, violation_series in checks.items():
            if violation_series.any():
                first_violation_idx = violation_series.idxmax()
                violation_data = df.loc[first_violation_idx, ['open', 'high', 'low', 'close']]
                raise ValueError(
                    f"Нарушение OHLC-инварианта '{description}'! "
                    f"Первое нарушение в индексе {first_violation_idx}:\n{violation_data}"
                )
 
    def _check_schema_and_order(self, df: pd.DataFrame):
        """
        Проверяет, что присутствуют ключевые колонки и что их порядок стабилен.
        Базовый префикс (строгий порядок):
          ['timestamp','symbol','open','high','low','close','volume','quote_asset_volume',
           'number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume']
        Остальные колонки допускаются после базового префикса.
        """
        prefix = [
            'timestamp','symbol','open','high','low','close','volume','quote_asset_volume',
            'number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume'
        ]
        missing = [c for c in prefix if c not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing}")
        # Проверяем порядок: первые len(prefix) колонок должны совпадать с prefix
        head = list(df.columns[:len(prefix)])
        if head != prefix:
            raise ValueError(f"Нарушен порядок колонок. Ожидается префикс {prefix}, получено {head}")

    def _check_no_pii(self, df: pd.DataFrame):
        """Проверяет отсутствие очевидных персональных данных в строковых колонках."""
        # Простые паттерны для e-mail и телефонов/SSN
        patterns = [
            re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
            re.compile(r"\b(?:\+?\d{1,3}[- ]?)?\d{3}[- ]?\d{3}[- ]?\d{4}\b"),
        ]
        object_cols = df.select_dtypes(include="object")
        for col in object_cols:
            series = object_cols[col].dropna().astype(str)
            for pattern in patterns:
                matches = series.str.contains(pattern, regex=True)
                if matches.any():
                    idx = matches.idxmax()
                    raise ValueError(
                        f"Обнаружены возможные персональные данные в колонке '{col}' по индексу {idx}: {series[idx]}"
                    )

    def _check_timestamp_continuity(self, df: pd.DataFrame, frequency: str = None):
        # если индекс не DateTimeIndex, проверяем равномерность шага
        if not isinstance(df.index, pd.DatetimeIndex):
            arr = df.index.to_numpy()

            if arr.size <= 1:
                return

            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(
                    "Неподдерживаемый тип индекса для проверки непрерывности. "
                    "Ожидается числовой индекс."
                )

            diffs = np.diff(arr)
            if diffs.size == 0:
                return

            non_zero = diffs[diffs != 0]
            if non_zero.size == 0:
                raise ValueError("Невозможно определить шаг индекса: все значения повторяются.")

            expected_step = non_zero[0]
            if (diffs != expected_step).any():
                raise ValueError("timestamp gap detected")
            return
        """Проверяет непрерывность временного ряда в DatetimeIndex."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Для проверки непрерывности индекс датафрейма должен быть pd.DatetimeIndex.")
        
        if not df.index.is_monotonic_increasing:
             raise ValueError("Нарушена монотонность временного ряда. Данные должны быть отсортированы по возрастанию времени.")

        if frequency is None:
            # Автоматическое определение частоты как моды разниц временных меток
            diffs = df.index.to_series().diff().dropna()
            if diffs.empty:
                # Если всего одна или ноль строк, разрывов нет
                return
            frequency = diffs.mode()[0]
            print(f"Автоматически определенная частота: {frequency}. Для принудительного задания используйте аргумент 'frequency'.")


        # Создание идеального временного ряда
        ideal_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=frequency)

        # Поиск пропущенных временных меток
        missing_timestamps = ideal_index.difference(df.index)

        if not missing_timestamps.empty:
            num_missing = len(missing_timestamps)
            # Показываем первые 10 пропусков для удобства
            preview = missing_timestamps[:10].tolist()
            raise ValueError(
                f"Обнаружен разрыв в непрерывности временного ряда (частота: {frequency}).\n"
                f"Всего пропущено таймстемпов: {num_missing}.\n"
                f"Примеры пропущенных таймстемпов: {preview}"
            )
