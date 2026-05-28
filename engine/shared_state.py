"""
Shared Runtime State Manager.

This module provides a centralized, thread-safe runtime state class 
that tracks query attributes, intermediate calculations, retrieved context, 
and pipeline execution trace.
"""

import time
import threading
from typing import Any, Dict, List, Optional


class SharedState:
    """
    Centralized execution state container.
    All modules read from and write to this state in a thread-safe manner.
    """

    def __init__(self, query: str = "", quantity: int = 100):
        self._lock = threading.Lock()
        
        self.query: str = query
        self.quantity: int = quantity
        self.workflow_plan: List[str] = []
        
        # Retrieval context
        self.retrieved_docs: List[str] = []
        self.vendor_profiles: List[Dict[str, Any]] = []
        
        # Calculators and business logic results
        self.costing_results: Dict[str, Any] = {}
        self.vendor_scores: List[Dict[str, Any]] = []
        self.compliance_results: Dict[str, Any] = {}
        self.web_search_results: List[Dict[str, Any]] = []
        self.structured_response: Dict[str, Any] = {}
        
        # Context/Memory
        self.memory_context: Dict[str, Any] = {
            "chat_history": [],
            "last_best_option": None
        }
        
        # Operational observability fields
        self.module_status: Dict[str, str] = {}  # module_name -> 'not_started'|'running'|'completed'|'failed'|'skipped'
        self.errors: List[Dict[str, Any]] = []  # [{"module": "...", "message": "...", "fatal": bool}]
        self.execution_metadata: Dict[str, Any] = {
            "start_time": time.time(),
            "end_time": None,
            "total_duration_ms": 0.0
        }
        
        # Logging and verification trace
        self.execution_trace: List[Dict[str, Any]] = []
        
        self.add_trace("State Initialized", {"query": query, "quantity": quantity})

    def add_trace(self, action: str, details: Optional[Dict[str, Any]] = None):
        """Append an execution step to the verification trace safely."""
        with self._lock:
            elapsed = (time.time() - self.execution_metadata["start_time"]) * 1000.0
            self.execution_trace.append({
                "timestamp": time.time(),
                "elapsed_ms": round(elapsed, 2),
                "action": action,
                "details": details or {}
            })

    def set_module_status(self, module_name: str, status: str):
        """Set the execution status of a specific module safely."""
        with self._lock:
            self.module_status[module_name] = status
            
    def add_error(self, module_name: str, message: str, fatal: bool = False):
        """Register an error within the shared state safely."""
        with self._lock:
            self.errors.append({
                "timestamp": time.time(),
                "module": module_name,
                "message": message,
                "fatal": fatal
            })
            
    def finalize_metadata(self):
        """Finalize the execution timings safely."""
        with self._lock:
            end_time = time.time()
            self.execution_metadata["end_time"] = end_time
            duration = (end_time - self.execution_metadata["start_time"]) * 1000.0
            self.execution_metadata["total_duration_ms"] = round(duration, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a JSON-serializable dictionary safely."""
        with self._lock:
            return {
                "query": self.query,
                "quantity": self.quantity,
                "workflow_plan": self.workflow_plan,
                "retrieved_docs": self.retrieved_docs,
                "vendor_profiles": self.vendor_profiles,
                "costing_results": self.costing_results,
                "vendor_scores": self.vendor_scores,
                "compliance_results": self.compliance_results,
                "web_search_results": self.web_search_results,
                "structured_response": self.structured_response,
                "memory_context": self.memory_context,
                "execution_trace": self.execution_trace,
                "module_status": self.module_status,
                "errors": self.errors,
                "execution_metadata": self.execution_metadata,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SharedState":
        """Reconstruct state object from a dictionary."""
        state = cls(query=data.get("query", ""), quantity=data.get("quantity", 100))
        with state._lock:
            state.workflow_plan = data.get("workflow_plan", [])
            state.retrieved_docs = data.get("retrieved_docs", [])
            state.vendor_profiles = data.get("vendor_profiles", [])
            state.costing_results = data.get("costing_results", {})
            state.vendor_scores = data.get("vendor_scores", [])
            state.compliance_results = data.get("compliance_results", {})
            state.web_search_results = data.get("web_search_results", [])
            state.structured_response = data.get("structured_response", {})
            state.memory_context = data.get("memory_context", {"chat_history": [], "last_best_option": None})
            state.execution_trace = data.get("execution_trace", [])
            state.module_status = data.get("module_status", {})
            state.errors = data.get("errors", [])
            state.execution_metadata = data.get("execution_metadata", {
                "start_time": time.time(),
                "end_time": None,
                "total_duration_ms": 0.0
            })
        return state

