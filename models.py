"""
formated data class for TVP pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

@dataclass
class Question:
    """a single question from the dataset"""
    image_index: str
    question_index: int
    image_filename: str
    question: str
    answer: Any
    answer_type: str

@dataclass
class Tool:
    """a predefined or learned tool/function"""
    name: str
    level: int  # 0 for predefined, higher for learned
    signature: str
    docstring: str
    implementation: str
    # source tracking (if abstracted)
    source_cluster_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    # deduplication tracking (if merged)
    merged_from: List[str] = field(default_factory=list)
    # deprecation tracking (false = merged into other tools)
    is_deprecated: bool = False
    deprecation_reason: Optional[str] = None
    usage_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'level': self.level,
            'signature': self.signature,
            'docstring': self.docstring,
            'implementation': self.implementation,
            'source_cluster_id': self.source_cluster_id,
            'created_at': self.created_at,
            'usage_count': self.usage_count,
            'merged_from': self.merged_from,
            'is_deprecated': self.is_deprecated,
            'deprecation_reason': self.deprecation_reason
        }

@dataclass
class Example:
    """an example in example library"""
    id: str
    question: Question
    program: str
    quality_rating: float
    execution_result: Any  # last updated execution result
    question_embedding: np.ndarray
    tools_used: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    iteration: int = 1  # last updated iter
    update_history: List[Dict[str, Any]] = field(default_factory=list)
    original_program: Optional[str] = None  # program replaced by last update
    original_execution_result: Optional[Any] = None # exe result replaced by last update
    last_updated: Optional[str] = None
    abstraction_status: str = "pending"  # "pending", "abstracted", "failed" 
    abstracted_tools: List[str] = field(default_factory=list)  # tools abstracted from this example
    abstraction_cluster_ids: List[str] = field(default_factory=list)  # clusters this example participated in

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'question_obj': self.question.__dict__,
            'question_index': self.question.question_index,
            'question': self.question.question,
            'answer': self.question.answer,
            'answer_type': self.question.answer_type,
            'program': self.program,
            'quality_rating': self.quality_rating,
            'execution_result': self.execution_result,
            'tools_used': self.tools_used,
            'created_at': self.created_at,
            'iteration': self.iteration,
            'update_history': self.update_history,
            'original_program': self.original_program,
            'original_execution_result': self.original_execution_result,
            'last_updated': self.last_updated,
            'abstraction_status': self.abstraction_status,
            'abstracted_tools': self.abstracted_tools,
            'abstraction_cluster_ids': self.abstraction_cluster_ids,
        }

@dataclass
class Cluster:
    """a cluster of similar examples (pending abstraction)"""
    id: str
    example_ids: List[str]
    pattern: str
    parameters: str
    abstraction_potential: float
    reasoning: str
    is_abstracted: bool = False
    abstracted_tool_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'example_ids': self.example_ids,
            'pattern': self.pattern,
            'parameters': self.parameters,
            'abstraction_potential': self.abstraction_potential,
            'reasoning': self.reasoning,
            'is_abstracted': self.is_abstracted,
            'abstracted_tool_names': self.abstracted_tool_names,
        }

@dataclass
class ProgramCandidate:
    """a candidate program during generation"""
    question: Question
    code: str
    executed: bool = False
    execution_result: Optional[Dict[str, Any]] = None
    execution_error: Optional[str] = None
    answer: Optional[Any] = None
    quality_rating: Optional[float] = None
    quality_reasoning: Optional[str] = None
    candidate_index: int = 0
    used_examples: List[str] = field(default_factory=list)

@dataclass
class ValidationResult:
    """results from tool validation"""
    tool_name: str
    cluster_id: str
    # stage 1: execution validation
    total_examples: int = 0
    execution_successes: int = 0
    execution_failures: int = 0
    execution_errors: List[Dict[str, Any]] = field(default_factory=list)
    # stage 2: correctness validation
    correctness_checks: int = 0
    correct_results: int = 0
    correctness_details: List[Dict[str, Any]] = field(default_factory=list)
    passed: bool = False    # overall result
    retry_count: int = 0
    successful_rewrites: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def execution_success_rate(self) -> float:
        if self.total_examples == 0:
            return 0.0
        return self.execution_successes / self.total_examples

    @property
    def correctness_rate(self) -> float:
        """correctness rate among divergent results only"""
        if self.correctness_checks == 0:
            return 1.0
        return self.correct_results / self.correctness_checks

    @property
    def overall_correctness_rate(self) -> float:
        """calculate correctness rate among all examples (non-divergent = correct)"""
        if self.total_examples == 0:
            return 0.0
        non_divergent_correct = self.execution_successes - self.correctness_checks
        total_correct = non_divergent_correct + self.correct_results
        return total_correct / self.total_examples

@dataclass
class AbstractionInterval:
    """tracks abstraction interval progress"""
    interval_number: int
    start_question_index: int
    end_question_index: int
    clusters_found: int = 0
    tools_created: int = 0
    validation_failures: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """convert to dictionary for serialization"""
        return {
            'interval_number': self.interval_number,
            'start_question_index': self.start_question_index,
            'end_question_index': self.end_question_index,
            'clusters_found': self.clusters_found,
            'tools_created': self.tools_created,
            'validation_failures': self.validation_failures,
            'started_at': self.started_at,
            'completed_at': self.completed_at
        }


@dataclass
class DeduplicationInterval:
    """a tool deduplication event"""
    interval_number: int
    start_question_index: int
    end_question_index: int
    tools_before: int = 0
    tools_after: int = 0
    eligible_tools: int = 0
    merge_groups_found: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'interval_number': self.interval_number,
            'start_question_index': self.start_question_index,
            'end_question_index': self.end_question_index,
            'tools_before': self.tools_before,
            'tools_after': self.tools_after,
            'eligible_tools': self.eligible_tools,
            'merge_groups_found': self.merge_groups_found,
            'started_at': self.started_at,
            'completed_at': self.completed_at
        }

@dataclass 
class ExampleUpdateRecord:
    """an example update"""
    example_id: str
    update_type: str
    old_program: str
    new_program: str
    old_tools_used: List[str]
    new_tools_used: List[str]
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    validation_passed: Dict[str, bool] = field(default_factory=dict)