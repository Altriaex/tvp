"""
config management for TVP ibrary building pipeline.
"""
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class PipelineConfig:    
    # dataset and paths
    dataset: str
    annotations_path: str
    image_path: str
    models_path: str
    results_path: str
    api_key_path: str
    # model selection for predefined apis
    vqa_model: str
    grounding_model: str
    grounding_model_box_threshold: float
    grounding_model_text_threshold: float
    depth_model: str
    # question loading and processing
    num_questions: int
    num_iterations: int
    programs_per_question: int
    max_workers: int
    random_seed: int
    reorder_questions_per_iteration: bool
    # example library and retrieval
    quality_threshold: float
    similarity_threshold: float
    max_retrieved_examples: int
    embedding_model: str
    # clustering and abstraction
    abstraction_interval: int
    clustering_threshold: float
    min_cluster_size: int
    abstraction_potential_threshold: float
    # validation
    min_execution_success_rate: float
    min_correctness_rate: float
    max_rewrite_retries: int
    max_abstraction_retries: int
    max_merge_retries: int
    # tool deduplication
    enable_deduplication: bool
    deduplication_interval: int
    deduplication_threshold: float
    min_tool_uses: int
    max_tool_uses: int
    # example update
    update_examples_after_abstraction: bool
    require_correctness_for_update: bool
    update_examples_after_deduplication: bool
    # model selection
    program_model: str
    judge_model: str
    cluster_model: str
    abstraction_model: str
    deduplication_model: str
    correctness_model: str
    rewrite_model: str
    merge_model: str
    # temperatures
    temperature_program: float
    temperature_judge: float
    temperature_abstraction: float
    temperature_cluster: float
    temperature_deduplication: float
    temperature_correctness: float
    temperature_rewrite: float
    temperature_merge: float
    reasoning_effort: str
    # utils
    resume_run_id: str
    debug: bool
    execution_timeout: int
    save_exe_trace: bool
    
    def __post_init__(self):
        self.run_id = self.resume_run_id
        self.results_dir = Path(self.results_path) / self.run_id
        
    def save(self, path):
        """save config"""
        config_dict = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path):
        """load config"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        config_dict.pop('results_dir', None)
        return cls(**config_dict)


def create_config_from_args(args) -> PipelineConfig:
    """create TVP config from command line args"""
    return PipelineConfig(
        # dataset and paths
        dataset=args.dataset,
        annotations_path=args.annotations_json,
        image_path=args.image_path,
        models_path=args.models_path,
        results_path=args.results_path,
        api_key_path=args.api_key_path,
        # model selection for predefined apis
        vqa_model=args.vqa_model,
        grounding_model=args.grounding_model,
        grounding_model_box_threshold=args.grounding_model_box_threshold,
        grounding_model_text_threshold=args.grounding_model_text_threshold,
        depth_model=args.depth_model,
        # question loading and processing
        num_questions=args.num_questions,
        num_iterations=args.num_iterations,
        programs_per_question=args.programs_per_question,
        max_workers=args.max_workers,
        random_seed=args.random_seed,
        reorder_questions_per_iteration=args.reorder_questions_per_iteration,
        # example library and retrieval
        quality_threshold=args.quality_threshold,
        similarity_threshold=args.similarity_threshold,
        max_retrieved_examples=args.max_retrieved_examples,
        embedding_model=args.embedding_model,
        # clustering and abstraction
        abstraction_interval=args.abstraction_interval,
        clustering_threshold=args.clustering_threshold,
        min_cluster_size=args.min_cluster_size,
        abstraction_potential_threshold=args.abstraction_potential_threshold,
        # validation
        min_execution_success_rate=args.min_execution_success_rate,
        min_correctness_rate=args.min_correctness_rate,
        max_rewrite_retries=args.max_rewrite_retries,
        max_abstraction_retries=args.max_abstraction_retries,
        max_merge_retries=args.max_merge_retries,
        # tool deduplication
        enable_deduplication=args.enable_deduplication,
        deduplication_interval=args.deduplication_interval,
        deduplication_threshold=args.deduplication_threshold,
        min_tool_uses=args.min_tool_uses,
        max_tool_uses=args.max_tool_uses,
        # example update
        update_examples_after_abstraction=args.update_examples_after_abstraction,
        require_correctness_for_update=args.require_correctness_for_update,
        update_examples_after_deduplication=args.update_examples_after_deduplication,
        # model selection
        program_model=args.program_model,
        judge_model=args.judge_model,
        cluster_model=args.cluster_model,
        abstraction_model=args.abstraction_model,
        deduplication_model=args.deduplication_model,
        correctness_model=args.correctness_model,
        rewrite_model=args.rewrite_model,
        merge_model=args.merge_model,
        # temperatures
        temperature_program=args.temperature_program,
        temperature_judge=args.temperature_judge,
        temperature_abstraction=args.temperature_abstraction,
        temperature_cluster=args.temperature_cluster,
        temperature_deduplication=args.temperature_deduplication,
        temperature_correctness=args.temperature_correctness,
        temperature_rewrite=args.temperature_rewrite,
        temperature_merge=args.temperature_merge,
        reasoning_effort=args.reasoning_effort,
        # utils
        resume_run_id=args.resume,
        debug=args.debug,
        execution_timeout=args.execution_timeout,
        save_exe_trace=args.save_exe_trace
    )
