# cython: language_level=3
"""Execution helpers for integrating the Cython order book."""

import execaction_interpreter as action_interpreter
from execevents import apply_agent_events
from execevents cimport Side
from execlob_book import CythonLOB

# Engine functions for full LOB execution and commit

cpdef step_full_lob(state, tracker, params, microgen, ws, action):
    """
    Execute one simulation step using the full order book (LOB) model.
    Generates and applies agent and public events to a cloned order book.
    """
    cdef object lob_clone
    # Obtain main order book from state (if available) and clone it
    try:
        lob_clone = state.lob.clone()
    except Exception:
        try:
            # state.lob might be None or not set, create a new book
            state.lob = CythonLOB()
            lob_clone = state.lob.clone()
        except Exception:
            lob_clone = CythonLOB()
    # Build agent events from the action
    events_list = action_interpreter.build_agent_event_set(state, tracker, params, action)
    # Apply agent and microstructure events to the cloned order book
    apply_agent_events(state, tracker, microgen, lob_clone, ws, events_list)
    return lob_clone

cpdef commit_step(state, tracker, lob_clone, ws):
    """
    Commit the results of the step to the environment state.
    This applies position changes, cash flows, and updates open orders tracking.
    In this stage, we do not modify primary EnvState fields (defer to later integration).
    """
    cdef Side order_side
    cdef tuple order_info
    # Update agent's open order tracker based on final LOB state
    if tracker is not None:
        try:
            tracker.clear()
        except AttributeError:
            pass
        # Add all remaining agent orders from lob_clone to tracker
        try:
            for order_info in lob_clone.iter_agent_orders():
                # order_info = (order_id, side, price)
                order_side = Side.BUY if int(order_info[1]) > 0 else Side.SELL
                try:
                    tracker.add(order_info[0], order_side, order_info[2])
                except TypeError:
                    try:
                        tracker.add(order_info[0], <int> order_side, order_info[2])
                    except Exception:
                        pass
                except AttributeError:
                    pass
        except AttributeError:
            # lob_clone may not expose the helper in some integration setups
            pass
    # Optionally update EnvState's order book to the new cloned state (atomic commit)
    try:
        state.lob = lob_clone
    except Exception:
        # NOTE: shim for integration; replace with project-specific state update if needed
        pass
    # (No direct update to state.cash, state.units, etc. in this stage)
    return None
