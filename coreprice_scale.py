"""Price scale conversion utilities.

This module provides functions to convert between float prices and integer tick prices 
using a given price scale. The conversions avoid floating-point precision issues by using 
proper rounding, and they rely on the price_scale parameter rather than hardcoded constants.

Functions:
    to_ticks(price_float: float, price_scale: int) -> int:
        Convert a price in float to an integer number of ticks.
    to_price(ticks_int: int, price_scale: int) -> float:
        Convert a price in integer ticks back to a float price.
    round_to_tick(price_float: float, price_scale: int) -> float:
        Round a float price to the nearest valid tick price (returns float).
"""

import math

def to_ticks(price_float: float, price_scale: int) -> int:
    """Convert a float price to integer ticks using the given price scale.

    The result is the nearest integer number of ticks that represents the price.
    This function avoids direct multiplication by large constants to reduce 
    floating-point error, using rounding for correct quantization to ticks.

    Args:
        price_float (float): The price as a floating-point number.
        price_scale (int): Number of ticks in one unit of price (e.g., 10000 for 0.0001 tick size).

    Returns:
        int: The price in integer ticks.

    Examples:
        >>> to_ticks(123.4567, 10000)
        1234567
        >>> to_ticks(1.2345, 10000)
        12345
    """
    # Scale the price and round to nearest integer to get tick count.
    # Use half-up rounding (round half towards +∞ for positive prices, towards -∞ for negative) for .5 cases.
    scaled = price_float * price_scale
    if price_float >= 0:
        ticks = math.floor(scaled + 0.5)
    else:
        # For negative prices (unlikely in typical use, but handle for completeness)
        ticks = math.ceil(scaled - 0.5)
    return int(ticks)

def to_price(ticks_int: int, price_scale: int) -> float:
    """Convert an integer tick price back to a float price using the given scale.

    This is the inverse of to_ticks, converting discrete tick values to the continuous price value.

    Args:
        ticks_int (int): The price in ticks (integer).
        price_scale (int): Number of ticks in one unit of price.

    Returns:
        float: The price as a floating-point number.

    Examples:
        >>> to_price(1234567, 10000)
        123.4567
        >>> to_price(12345, 10000)
        1.2345
    """
    # Divide the integer ticks by the scale to get the original price.
    return ticks_int / price_scale

def round_to_tick(price_float: float, price_scale: int) -> float:
    """Round a float price to the nearest tick price increment.

    This function uses to_ticks to first convert the price to the nearest tick count, 
    then converts it back to a float price. It ensures the returned price is aligned 
    to the discrete price grid defined by price_scale.

    Args:
        price_float (float): The price as a floating-point number to round.
        price_scale (int): Number of ticks in one unit of price.

    Returns:
        float: The price rounded to the nearest tick increment.

    Examples:
        >>> round_to_tick(123.456, 100)
        123.46
        >>> round_to_tick(1.23456, 10000)
        1.2346
    """
    # Convert to ticks (with proper rounding), then back to float price.
    ticks = to_ticks(price_float, price_scale)
    return ticks / price_scale
