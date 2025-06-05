"""
Verification models for the chain framework.
"""
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union, Set
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import uuid


class VerificationStatus(str, Enum):
    """Status of verification tasks and chains."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"


class VerificationPassType(str, Enum):
    """Types of verification passes."""
    CLAIM_EXTRACTION = "claim_extraction"
    CITATION_CHECK = "citation_check"
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    LOGIC_ANALYSIS = "logic_analysis"
    BIAS_SCAN = "bias_scan"
    ADVERSARIAL_VALIDATION = "adversarial_validation"


class Priority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Pydantic models for API serialization
class VerificationPassConfig(BaseModel):
    """Configuration for a single verification pass."""
    pass_type: VerificationPassType
    pass_id: str = Field(description="Unique identifier for this pass instance")
    name: str
    description: Optional[str] = None
    enabled: bool = True
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: int = 10
    parameters: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[VerificationPassType] = Field(
        default_factory=list,
        description="List of pass types that must complete before this pass can run"
    )
    
    @validator('pass_id')
    def validate_pass_id(cls, v):
        """Ensure pass_id is not empty and is a valid identifier."""
        if not v or not v.strip():
            raise ValueError("pass_id cannot be empty")
        # Ensure it's a valid identifier (alphanumeric, underscore, hyphen)
        if not all(c.isalnum() or c in '_-' for c in v):
            raise ValueError("pass_id must contain only alphanumeric characters, underscores, and hyphens")
        return v.strip()
    
    @validator('depends_on')
    def validate_dependencies(cls, v, values):
        """Ensure dependencies don't include self-reference."""
        if 'pass_type' in values and values['pass_type'] in v:
            raise ValueError("A pass cannot depend on itself")
        return v


class VerificationChainConfig(BaseModel):
    """Configuration for a complete verification chain."""
    chain_id: str
    name: str
    description: Optional[str] = None
    passes: List[VerificationPassConfig]
    global_timeout_seconds: int = 3600
    parallel_execution: bool = False
    stop_on_failure: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('passes')
    def validate_passes(cls, v):
        """Validate passes configuration and dependencies."""
        if not v:
            raise ValueError("At least one pass must be configured")
        
        # Check for duplicate pass_ids
        pass_ids = [pass_config.pass_id for pass_config in v]
        if len(pass_ids) != len(set(pass_ids)):
            duplicates = [pid for pid in pass_ids if pass_ids.count(pid) > 1]
            raise ValueError(f"Duplicate pass_ids found: {duplicates}")
        
        # Get all available pass types in this chain
        available_pass_types = {pass_config.pass_type for pass_config in v}
        
        # Validate dependencies exist in chain
        for pass_config in v:
            for dependency in pass_config.depends_on:
                if dependency not in available_pass_types:
                    raise ValueError(
                        f"Pass '{pass_config.pass_id}' depends on '{dependency}' "
                        f"which is not present in this chain"
                    )
        
        # Check for circular dependencies
        cls._check_circular_dependencies(v)
        
        return v
    
    @staticmethod
    def _check_circular_dependencies(passes: List[VerificationPassConfig]):
        """Check for circular dependencies using depth-first search."""
        # Build dependency graph: pass_type -> set of dependent pass_types
        dependency_graph = {}
        for pass_config in passes:
            dependency_graph[pass_config.pass_type] = set(pass_config.depends_on)
        
        # Track visit states: 0=unvisited, 1=visiting, 2=visited
        visit_state = {pass_type: 0 for pass_type in dependency_graph.keys()}
        
        def dfs(pass_type: VerificationPassType, path: List[VerificationPassType]):
            if visit_state[pass_type] == 1:  # Currently visiting - cycle detected
                cycle_start = path.index(pass_type)
                cycle = path[cycle_start:] + [pass_type]
                cycle_str = " -> ".join(pt.value for pt in cycle)
                raise ValueError(f"Circular dependency detected: {cycle_str}")
            
            if visit_state[pass_type] == 2:  # Already visited
                return
            
            visit_state[pass_type] = 1  # Mark as visiting
            
            for dependency in dependency_graph[pass_type]:
                dfs(dependency, path + [pass_type])
            
            visit_state[pass_type] = 2  # Mark as visited
        
        # Check each pass type for cycles
        for pass_type in dependency_graph.keys():
            if visit_state[pass_type] == 0:
                dfs(pass_type, [])
    
    def get_sequential_execution_order(self) -> List[VerificationPassConfig]:
        """Get passes in dependency-resolved sequential execution order using topological sort."""
        return self._get_sequential_execution_order()
    
    def get_parallel_execution_groups(self) -> List[List[VerificationPassConfig]]:
        """Get passes grouped by dependency levels for parallel execution."""
        return self._get_parallel_execution_groups()
    
    def get_execution_order(self) -> List[VerificationPassConfig]:
        """
        Get passes in dependency-resolved execution order.
        
        Note: This method only returns sequential order. For parallel execution,
        use get_parallel_execution_groups() instead.
        
        Returns:
            List of passes in sequential execution order
        """
        return self._get_sequential_execution_order()
    
    def _get_sequential_execution_order(self) -> List[VerificationPassConfig]:
        """Get passes in sequential execution order (topological sort)."""
        # Build maps for quick lookup
        pass_by_type = {p.pass_type: p for p in self.passes}
        
        # Calculate in-degree for each pass
        in_degree = {p.pass_type: 0 for p in self.passes}
        for pass_config in self.passes:
            for dependency in pass_config.depends_on:
                in_degree[pass_config.pass_type] += 1
        
        # Start with passes that have no dependencies
        queue = [pass_type for pass_type, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort by pass type for deterministic ordering
            queue.sort()
            current_pass_type = queue.pop(0)
            result.append(pass_by_type[current_pass_type])
            
            # Reduce in-degree for passes that depend on current pass
            for pass_config in self.passes:
                if current_pass_type in pass_config.depends_on:
                    in_degree[pass_config.pass_type] -= 1
                    if in_degree[pass_config.pass_type] == 0:
                        queue.append(pass_config.pass_type)
        
        return result
    
    def _get_parallel_execution_groups(self) -> List[List[VerificationPassConfig]]:
        """Get passes grouped by dependency levels for parallel execution."""
        pass_by_type = {p.pass_type: p for p in self.passes}
        groups = []
        remaining_passes = set(p.pass_type for p in self.passes)
        completed_passes = set()
        
        while remaining_passes:
            # Find passes that can run now (all dependencies satisfied)
            ready_passes = []
            for pass_type in remaining_passes:
                pass_config = pass_by_type[pass_type]
                if all(dep in completed_passes for dep in pass_config.depends_on):
                    ready_passes.append(pass_config)
            
            if not ready_passes:
                # This shouldn't happen if circular dependency check passed
                raise ValueError("Unable to resolve execution order - possible circular dependency")
            
            groups.append(ready_passes)
            for pass_config in ready_passes:
                remaining_passes.remove(pass_config.pass_type)
                completed_passes.add(pass_config.pass_type)
        
        return groups


class VerificationResult(BaseModel):
    """Result from a single verification pass."""
    pass_id: str = Field(description="Unique identifier for the pass instance")
    pass_type: VerificationPassType
    status: VerificationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    result_data: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerificationChainResult(BaseModel):
    """Result from a complete verification chain execution."""
    chain_id: str
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    status: VerificationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_execution_time_seconds: Optional[float] = None
    pass_results: List[VerificationResult] = Field(default_factory=list)
    overall_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    summary: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class VerificationTask(BaseModel):
    """A verification task to be processed."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    chain_config: VerificationChainConfig
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    status: VerificationStatus = VerificationStatus.PENDING
    result: Optional[VerificationChainResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Dataclasses for internal processing
@dataclass
class VerificationContext:
    """Context information passed between verification passes."""
    document_id: str
    document_content: str
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    chain_config: Optional[VerificationChainConfig] = None
    previous_results: List[VerificationResult] = field(default_factory=list)
    global_context: Dict[str, Any] = field(default_factory=dict)
    
    def get_previous_result(self, pass_type: VerificationPassType) -> Optional[VerificationResult]:
        """Get the result from a previous pass by type."""
        for result in self.previous_results:
            if result.pass_type == pass_type:
                return result
        return None
    
    def get_result_by_id(self, pass_id: str) -> Optional[VerificationResult]:
        """Get the result from a previous pass by its unique pass_id."""
        for result in self.previous_results:
            if result.pass_id == pass_id:
                return result
        return None
    
    def get_all_results_by_type(self, pass_type: VerificationPassType) -> List[VerificationResult]:
        """Get all results from a specific pass type."""
        return [result for result in self.previous_results if result.pass_type == pass_type]
    
    def get_completed_pass_types(self) -> Set[VerificationPassType]:
        """Get set of all completed pass types for dependency checking."""
        return {
            result.pass_type for result in self.previous_results 
            if result.status == VerificationStatus.COMPLETED
        }
    
    def are_dependencies_satisfied(self, pass_config: 'VerificationPassConfig') -> bool:
        """Check if all dependencies for a pass are satisfied."""
        completed_types = self.get_completed_pass_types()
        return all(dep_type in completed_types for dep_type in pass_config.depends_on)


@dataclass 
class PassExecutionConfig:
    """Configuration for executing a verification pass."""
    pass_config: VerificationPassConfig
    context: VerificationContext
    max_retries: int = 3
    timeout_seconds: int = 300
    retry_delay_seconds: int = 10


# Exception classes
class VerificationError(Exception):
    """Base exception for verification errors."""
    pass


class VerificationTimeoutError(VerificationError):
    """Raised when verification pass times out."""
    pass


class VerificationConfigError(VerificationError):
    """Raised when there's an error in verification configuration."""
    pass


class PassDependencyError(VerificationError):
    """Raised when pass dependencies are not satisfied."""
    pass


# Metrics and monitoring
class VerificationMetrics(BaseModel):
    """Metrics for verification performance monitoring."""
    total_tasks_processed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_execution_time_seconds: float = 0.0
    pass_success_rates: Dict[str, float] = Field(default_factory=dict)
    error_frequencies: Dict[str, int] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))