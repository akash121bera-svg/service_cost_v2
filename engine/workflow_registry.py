"""
Centralized Workflow Registry.

Decouples processing tools from the Orchestrator loop by mapping stage keys 
to executable wrapper functions. Provides dependency declarations, validation
contracts, and execution parameters.
"""

from typing import Dict, Any, Callable, List, Optional

# Import all executable modules
from engine.memory import run_memory_layer
from engine.uploaded_costs import run_costing_engine, run_vendor_logic
from engine.compliance import run_compliance_checks
from engine.search_enrichment import run_search_enrichment
from rag.pipeline import retrieve_context


class WorkflowRegistry:
    """
    Registry container mapping stage IDs to runnable wrappers and metadata.
    """

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._register_default_workflows()

    def register(
        self,
        stage_id: str,
        func: Callable,
        stage_num: int,
        parallel_safe: bool = False,
        dependencies: Optional[List[str]] = None
    ):
        """
        Register a pipeline workflow function.
        """
        self._registry[stage_id] = {
            "func": func,
            "stage_num": stage_num,
            "parallel_safe": parallel_safe,
            "dependencies": dependencies or []
        }

    def get_function(self, stage_id: str) -> Optional[Callable]:
        """
        Retrieve execution handler for the given stage.
        """
        workflow = self._registry.get(stage_id)
        return workflow["func"] if workflow else None

    def get_metadata(self, stage_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata parameters of the registered workflow.
        """
        return self._registry.get(stage_id)

    def validate_plan(self, plan: List[str]) -> List[str]:
        """
        Verify that a list of scheduled stages satisfies dependency constraints.
        Resolves missing prerequisites automatically.
        """
        validated_plan = []
        for step in plan:
            meta = self.get_metadata(step)
            if not meta:
                continue
                
            # Resolve prerequisites
            for dep in meta["dependencies"]:
                if dep not in validated_plan:
                    validated_plan.append(dep)
                    
            if step not in validated_plan:
                validated_plan.append(step)
                
        return validated_plan

    def _register_default_workflows(self):
        """
        Set up the core Hybrid Workflow orchestration mapping.
        """
        from typing import Optional
        # Stage 1: Inbound context gathering (Sequential / Parallel)
        self.register(
            "MEM_LOAD",
            run_memory_layer,
            stage_num=1,
            parallel_safe=False,
            dependencies=[]
        )
        self.register(
            "RAG_RETRIEVAL",
            retrieve_context,
            stage_num=1,
            parallel_safe=False,
            dependencies=[]
        )
        self.register(
            "WEB_SEARCH",
            run_search_enrichment,
            stage_num=1,
            parallel_safe=False,
            dependencies=[]
        )
        
        # Stage 2: Parallel safe calculation, scoring, and regulatory audits
        self.register(
            "COSTING",
            run_costing_engine,
            stage_num=2,
            parallel_safe=True,
            dependencies=[]
        )
        self.register(
            "VENDOR_LOGIC",
            run_vendor_logic,
            stage_num=2,
            parallel_safe=True,
            dependencies=["COSTING"]
        )
        self.register(
            "COMPLIANCE",
            run_compliance_checks,
            stage_num=2,
            parallel_safe=True,
            dependencies=[]
        )


# Global workflow registry instance
WORKFLOW_REGISTRY = WorkflowRegistry()
