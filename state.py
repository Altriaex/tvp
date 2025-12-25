"""
state management for TVP pipeline run.
"""
import json
from datetime import datetime
import numpy as np

from models import Question, Tool, Example, Cluster, AbstractionInterval, DeduplicationInterval
from logging_utils import get_logger as get_logger_impl
from utils import EmbeddingManager

class PipelineState:
    """manages TVP pipeline run state for persistence and resumes"""
    def __init__(self, config):
        self.config = config
        self.state_dir = config.results_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # initialize
        self.current_iteration = 1
        self.questions_processed = 0
        self.abstraction_intervals_completed = 0
        self.deduplication_rounds = 0
        self.examples = []
        self.tools = []
        self.clusters = []
        self.abstraction_intervals = []
        self.deduplication_intervals = []
        self.embedding_manager = EmbeddingManager(config)   # safe fallback for corrupted embeddings
        # load existing state (resuming or keep init)
        if config.resume_run_id:
            self.load()
    
    def save(self):
        checkpoint = {
            'current_iteration': self.current_iteration,
            'questions_processed': self.questions_processed,
            'abstraction_intervals_completed': self.abstraction_intervals_completed,
            'deduplication_rounds': self.deduplication_rounds,
            'timestamp': get_timestamp()
        }
        with open(self.state_dir / "checkpoint.json", 'w') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        self._save_examples()
        self._save_tools()
        self._save_clusters()
        self._save_abstraction_intervals()
        self._save_deduplication_intervals()
        get_logger(__name__).debug("State saved successfully")
    
    def load(self):
        checkpoint_path = self.state_dir / "checkpoint.json"
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
            self.current_iteration = checkpoint.get('current_iteration', 1)
            self.questions_processed = checkpoint.get('questions_processed', 0)
            self.abstraction_intervals_completed = checkpoint.get('abstraction_intervals_completed', 0)
            self.deduplication_rounds = checkpoint.get('deduplication_rounds', 0)
            get_logger(__name__).info(f"Loaded checkpoint - Iteration: {self.current_iteration}, Questions: {self.questions_processed}")
        self._load_examples()
        self._load_tools()
        self._load_clusters()
        self._load_abstraction_intervals()
        self._load_deduplication_intervals()
    
    def _save_examples(self):
        examples_path = self.state_dir / "examples.json"
        examples_data = []
        for example in self.examples:
            examples_data.append(example.to_dict())
        with open(examples_path, 'w') as f:
            json.dump(examples_data, f, indent=2, ensure_ascii=False)
        embeddings_path = self.state_dir / "embeddings.npz" # save computed question embeddings
        embeddings = {
            f"{ex.id}_question": ex.question_embedding
            for ex in self.examples
        }
        np.savez_compressed(embeddings_path, **embeddings)
    
    def _load_examples(self):
        examples_path = self.state_dir / f"examples.json"
        embeddings_path = self.state_dir / "embeddings.npz"
        if examples_path.exists():
            with open(examples_path) as f:
                examples_data = json.load(f)
            embeddings = {}
            if embeddings_path.exists():
                embeddings = np.load(embeddings_path)   # load already computed question embeddings
            self.examples = []
            for ex_data in examples_data:
                if 'question_obj' in ex_data:
                    question_data = ex_data['question_obj']
                else:
                    question_data = {
                        'image_index': ex_data.get('image_index', ''),
                        'question_index': ex_data['question_index'],
                        'image_filename': ex_data.get('image_filename', ''),
                        'question': ex_data['question'],
                        'answer': ex_data['answer'],
                        'answer_type': ex_data['answer_type'],
                        'complexity_score': ex_data.get('complexity_score'),
                        'complexity_reasoning': ex_data.get('complexity_reasoning')
                    }
                question = Question(**question_data)
                question_embedding = embeddings.get(f"{ex_data['id']}_question")
                if question_embedding is None:  # safe fallback for corrupted embeddings
                    get_logger(__name__).warning(f"Loading example question_{ex_data['id']} embedding failed, redo the embedding now...")
                    question_embedding = self.embedding_manager.embed_text(question.question)
                example = Example(
                    id=ex_data['id'],
                    question=question,
                    program=ex_data['program'],
                    quality_rating=ex_data['quality_rating'],
                    execution_result=ex_data.get('execution_result'),
                    question_embedding=question_embedding,
                    tools_used=ex_data.get('tools_used', []),
                    created_at=ex_data.get('created_at', datetime.now().isoformat()),
                    iteration=ex_data.get('iteration', 1),
                    update_history=ex_data.get('update_history', []),
                    original_program=ex_data.get('original_program'),
                    original_execution_result=ex_data.get('original_execution_result'),
                    last_updated=ex_data.get('last_updated'),
                    abstraction_status=ex_data.get('abstraction_status', 'pending'),
                    abstracted_tools=ex_data.get('abstracted_tools', []),
                    abstraction_cluster_ids=ex_data.get('abstraction_cluster_ids', [])
                )
                self.examples.append(example)
            get_logger(__name__).info(f"Loaded {len(self.examples)} examples")
    
    def _save_tools(self):
        tools_path = self.state_dir / "tools.json"
        tools_data = []
        for tool in self.tools:
            tool_dict = tool.to_dict()
            tools_data.append(tool_dict)
        with open(tools_path, 'w') as f:
            json.dump(tools_data, f, indent=2, ensure_ascii=False)
    
    def _load_tools(self):
        tools_path = self.state_dir / "tools.json"
        if tools_path.exists():
            with open(tools_path) as f:
                tools_data = json.load(f)
            self.tools = []
            for tool_data in tools_data:
                tool = Tool(**tool_data)
                self.tools.append(tool)
            active_count = len([t for t in self.tools if not t.is_deprecated])
            deprecated_count = len([t for t in self.tools if t.is_deprecated])
            get_logger(__name__).info(f"Loaded {active_count} active tools and {deprecated_count} deprecated tools")
    
    def _save_clusters(self):
        clusters_path = self.state_dir / "clusters.json"
        clusters_data = []
        for cluster in self.clusters:
            clusters_data.append(cluster.to_dict())
        with open(clusters_path, 'w') as f:
            json.dump(clusters_data, f, indent=2, ensure_ascii=False)
    
    def _load_clusters(self):
        clusters_path = self.state_dir / "clusters.json"
        if clusters_path.exists():
            with open(clusters_path) as f:
                clusters_data = json.load(f)
            self.clusters = []
            for cluster_data in clusters_data:
                cluster = Cluster(**cluster_data)
                self.clusters.append(cluster)
            get_logger(__name__).info(f"Loaded {len(self.clusters)} clusters")
    
    def _save_abstraction_intervals(self):
        intervals_path = self.state_dir / "abstraction_intervals.json"
        intervals_data = []
        for interval in self.abstraction_intervals:
            intervals_data.append(interval.to_dict())
        with open(intervals_path, 'w') as f:
            json.dump(intervals_data, f, indent=2, ensure_ascii=False)
    
    def _load_abstraction_intervals(self):
        intervals_path = self.state_dir / "abstraction_intervals.json"
        if intervals_path.exists():
            with open(intervals_path) as f:
                intervals_data = json.load(f)
            self.abstraction_intervals = []
            for interval_data in intervals_data:
                interval = AbstractionInterval(**interval_data)
                self.abstraction_intervals.append(interval)
            get_logger(__name__).info(f"Loaded {len(self.abstraction_intervals)} abstraction intervals")
    
    def _save_deduplication_intervals(self):
        intervals_path = self.state_dir / "deduplication_intervals.json"
        intervals_data = []
        for interval in self.deduplication_intervals:
            intervals_data.append(interval.to_dict())
        with open(intervals_path, 'w') as f:
            json.dump(intervals_data, f, indent=2, ensure_ascii=False)
    
    def _load_deduplication_intervals(self):
        intervals_path = self.state_dir / "deduplication_intervals.json"
        if intervals_path.exists():
            with open(intervals_path) as f:
                intervals_data = json.load(f)
            self.deduplication_intervals = []
            for interval_data in intervals_data:
                interval = DeduplicationInterval(**interval_data)
                self.deduplication_intervals.append(interval)
            get_logger(__name__).info(f"Loaded {len(self.deduplication_intervals)} deduplication intervals")

def get_timestamp():
    return datetime.now().isoformat()

def get_logger(name):   
    return get_logger_impl(name)