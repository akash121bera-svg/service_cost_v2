"""
Memory Layer.

Provides lightweight conversation memory and continuity helpers,
tracking previous vendor options and chat logs within the shared state.
"""

from typing import Any, Dict, List, Optional
from engine.shared_state import SharedState


def run_memory_layer(state: SharedState, session_history: Optional[List[Dict[str, str]]] = None, last_best_option: Optional[Dict[str, Any]] = None):
    """
    Executes the memory layer, synchronizing session history 
    and tracking previously calculated best options.
    """
    if session_history:
        state.memory_context["chat_history"] = session_history
    
    if last_best_option:
        state.memory_context["last_best_option"] = last_best_option

    state.add_trace("Memory Layer Loaded", {
        "history_length": len(state.memory_context.get("chat_history", [])),
        "has_last_best_option": state.memory_context.get("last_best_option") is not None
    })


def update_best_option(state: SharedState, best_option: Dict[str, Any]):
    """Update the recorded best option in both state and memory."""
    state.memory_context["last_best_option"] = best_option
    state.add_trace("Memory Updated: Best Option Recorded", {"vendor": best_option.get("file")})
