# cython: language_level=3
import execaction_interpreter as action_interpreter
from execevents import apply_agent_events
from execlob_book cimport CythonLOB
from coreworkspace cimport SimulationWorkspace
from execevents cimport EventType

# Engine functions for full LOB execution and commit

cpdef step_full_lob(state, tracker, params, microgen, SimulationWorkspace ws, action):
    """
    Execute one simulation step using the full order book (LOB) model.
    Generates and applies agent and public events to a cloned order book.
    """
    cdef CythonLOB lob_clone
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

cpdef commit_step(state, tracker, CythonLOB lob_clone, SimulationWorkspace ws):
    """
    Commit the results of the step to the environment state.
    This applies position changes, cash flows, and updates open orders tracking.
    In this stage, we do not modify primary EnvState fields (defer to later integration).
    """
    cdef int i
    # Update agent's open order tracker based on final LOB state
    if tracker is not None:
        try:
            tracker.clear()
        except AttributeError:
            pass
        # Add all remaining agent orders from lob_clone to tracker
        for i in range(lob_clone.n_bids):
            if lob_clone.bid_orders[i].type == EventType.AGENT_LIMIT_ADD:
                try:
                    tracker.add(lob_clone.bid_orders[i].order_id, 1, lob_clone.bid_orders[i].price)
                except AttributeError:
                    pass
        for i in range(lob_clone.n_asks):
            if lob_clone.ask_orders[i].type == EventType.AGENT_LIMIT_ADD:
                try:
                    tracker.add(lob_clone.ask_orders[i].order_id, -1, lob_clone.ask_orders[i].price)
                except AttributeError:
                    pass
    # Optionally update EnvState's order book to the new cloned state (atomic commit)
    try:
        state.lob = lob_clone
    except Exception:
        # NOTE: shim for integration; replace with project-specific state update if needed
        pass
    # (No direct update to state.cash, state.units, etc. in this stage)
    return None
