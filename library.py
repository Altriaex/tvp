"""
example and tool libraries in TVP.
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from agents import QualityJudge
from models import ProgramCandidate, ExampleUpdateRecord
from utils import EmbeddingManager, extract_tools_from_program
from logging_utils import get_logger, log_example_added, log_retrieval, log_example_updated, log_tool_deprecation

class ExampleLibrary:
    """library of program examples"""
    def __init__(self, config):
        self.config = config
        self.examples = []
        self.embedding_manager = EmbeddingManager(config)
        self._id_to_example = {}
        self._question_idx_to_example = {}  # map question index to example
        self.update_records = []
        self.logger = get_logger(__name__)
        self._question_embedding_cache = {} # cache for question embeddings
        self.quality_judge = None   # for quality re-evaluation
    
    def set_tool_library(self, tool_library):
        """connect to tool library"""
        self.tool_library = tool_library
    
    def get_example(self, example_id):
        """get example by id"""
        return self._id_to_example.get(example_id)
    
    def retrieve_similar(self, query, max_examples=3, threshold=0.8, exclude_question_index=None):
        if not self.examples:
            return [], []
        if query in self._question_embedding_cache:
            query_embedding = self._question_embedding_cache[query]
            self.logger.debug(f"Using cached embedding for question: {query}...")
        else:
            query_embedding = self.embedding_manager.embed_text(query)
            self._cache_question_embedding(query, query_embedding)
        similarities = []
        for example in self.examples:
            if exclude_question_index is not None and example.question.question_index == exclude_question_index:
                continue
            sim = self._cosine_similarity(query_embedding, example.question_embedding)
            similarities.append((example, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        self.logger.debug(f"Query: {query} - similarities top 2: {similarities[:2]}...")
        retrieved = []
        scores = []
        for example, sim in similarities:
            if sim >= threshold and len(retrieved) < max_examples:
                retrieved.append(example)
                scores.append(sim)
        if retrieved:
            log_retrieval(
                question_id=exclude_question_index if exclude_question_index is not None else 0,
                retrieved_count=len(retrieved),
                similarities=scores
            )
        return retrieved, scores
    
    def _cache_question_embedding(self, question, embedding):
        self._question_embedding_cache[question] = embedding
    
    def _cosine_similarity(self, vec1, vec2):
        return float(np.dot(vec1, vec2))
    
    def add_example(self, example):
        """add a new example or update an existing example with new solution in later iteration"""
        existing_example = self._question_idx_to_example.get(example.question.question_index)
        if not existing_example:    # new example
            example.id = f"q{example.question.question_index}"
            self.examples.append(example)
            self._id_to_example[example.id] = example
            self._question_idx_to_example[example.question.question_index] = example
            self._cache_question_embedding(example.question.question, example.question_embedding)   # cache the question embedding
            log_example_added(
                example.id,
                example.quality_rating,
                example.tools_used
            )
        else:   # this question already exists --> updating across iterations
            self.logger.info(f"Question {example.question.question_index} already exists in library")
            need_quality_reevaluation = (
                existing_example.original_program is not None and
                existing_example.abstraction_status == "abstracted"  # has been abstracted (so the program rewritten with abstracted tool has not yet been evaluated) | otherwise --> just use existing quality rating stored in the lib)
            )
            existing_quality = existing_example.quality_rating  # lib-stored quality rating of existing example
            new_quality = example.quality_rating
            if need_quality_reevaluation:
                self.logger.info(f"Re-evaluating quality for existing abstracted example")
                reevaluated_quality = self._reevaluate_quality(existing_example, example.iteration)
                if reevaluated_quality is not None:
                    existing_quality = reevaluated_quality
                    existing_example.quality_rating = reevaluated_quality
                    self.logger.info(f"Re-evaluated quality: {reevaluated_quality}")
            self.logger.info(f"Quality comparison - Existing: {existing_quality}, New: {new_quality}")
            if new_quality > existing_quality:  # update if new is better
                self.logger.info(f"📝 Updating example for question {example.question.question_index} with new iteration solution")
                self._update_example_with_new_iteration(existing_example, example)
                log_example_updated(
                    example_id=existing_example.id,
                    update_type='iteration_update',
                    reason=f'New iteration solution (quality: {new_quality} > {existing_quality})',
                    old_tools=existing_example.tools_used,
                    new_tools=example.tools_used
                )
            elif new_quality == existing_quality:   # when qualities are equal, check if tools are the same
                existing_tools = set(existing_example.tools_used)
                new_tools = set(example.tools_used)
                if existing_tools != new_tools: # new tools used --> prefer new solution
                    self.logger.info(f"📝 Updating example with equal quality ({new_quality}) but different tools")
                    self.logger.info(f"  Existing tools: {existing_tools}")
                    self.logger.info(f"  New tools: {new_tools}")
                    self._update_example_with_new_iteration(existing_example, example)
                    log_example_updated(
                        example_id=existing_example.id,
                        update_type='iteration_update',
                        reason=f'New iteration solution with different tools (quality: {new_quality} == {existing_quality})',
                        old_tools=existing_example.tools_used,
                        new_tools=example.tools_used
                    )
                else:
                    self.logger.info(f"Keeping existing solution (same quality {existing_quality} and same tools)")
            else:
                self.logger.info(f"Keeping existing solution (quality: {existing_quality} > {new_quality})")

    def _reevaluate_quality(self, example, iteration):
        """re-evaluate quality of an example using quality judge"""
        if self.quality_judge is None:
            self.quality_judge = QualityJudge(self.config)
        try:
            if isinstance(example.execution_result, dict):
                answer = example.execution_result.get('final_result')
            else:
                answer = example.execution_result
            candidate = ProgramCandidate(
                question=example.question,
                code=example.program,
                execution_result=example.execution_result,
                execution_error=None,
                answer=answer,
                candidate_index=0
            )
            tools = []
            if hasattr(self, 'tool_library'):
                tools = self.tool_library.get_all_tools()
            output_dir = Path(self.config.results_dir) / "quality_reevaluation" / f"iteration_{iteration}" / example.id
            output_dir.mkdir(parents=True, exist_ok=True)
            judged_candidate = self.quality_judge.judge_single({
                'candidate': candidate,
                'output_dir': output_dir,
                'tools': tools
            })
            if judged_candidate and judged_candidate.quality_rating is not None:
                return judged_candidate.quality_rating
        except Exception as e:
            self.logger.error(f"Failed to re-evaluate quality: {e}")
        return None

    def _update_example_with_new_iteration(self, existing, new):
        """update the entry of existing example with new iteration's solution"""
        if existing.original_program is None:   # store original if first update
            existing.original_program = existing.program
            existing.original_execution_result = existing.execution_result
        # 1. record current entry
        old_program = existing.program
        old_execution_result = existing.execution_result
        old_quality = existing.quality_rating
        # 2. update fields
        existing.program = new.program
        existing.quality_rating = new.quality_rating
        existing.execution_result = new.execution_result
        existing.tools_used = new.tools_used
        existing.iteration = new.iteration
        existing.last_updated = datetime.now().isoformat()
        # 3. reset abstraction status --> eligible for clustering again (because it's a new program solution)
        existing.abstraction_status = "pending"
        # 4. add to update history
        existing.update_history.append({
            'timestamp': datetime.now().isoformat(),
            'update_type': 'iteration_update',
            'reason': f'Updated from iteration {new.iteration}',
            'old_quality': old_quality,
            'new_quality': new.quality_rating,
            'validation_passed': {'execution': True, 'correctness': True},
            'old_program': old_program,
            'new_program': new.program,
            'old_execution_result': old_execution_result,
            'new_execution_result': new.execution_result
        })

    def update_examples_for_new_tool(self, tool_name, successful_rewrites, update_type='abstraction'):
        """update multiple examples to use a new tool (after abstraction or tool merge)"""
        updated_count = 0
        for rewrite in successful_rewrites:
            example_id = rewrite['example_id']
            new_program = rewrite['new_program']
            validation_passed = rewrite.get('validation_passed', {'execution': True})
            new_execution_result = rewrite.get('new_namespace')
            if new_execution_result is None:
                new_execution_result = rewrite.get('new_answer')
            new_tools_used = extract_tools_from_program(new_program)
            if update_type == 'abstraction':
                reason = f"Tool abstraction: {tool_name}"
            elif update_type == 'deduplication':
                reason = f"Tool deduplication: merged into {tool_name}"
            else:
                reason = f"Tool update: {tool_name}"
            if self.update_example_program(
                example_id=example_id,
                new_program=new_program,
                new_tools_used=new_tools_used,
                reason=reason,
                update_type=update_type,
                validation_passed=validation_passed,
                new_execution_result=new_execution_result
            ):
                updated_count += 1
        self.logger.info(f"Updated {updated_count} examples for tool {tool_name}")
        return updated_count
    
    def update_example_program(self, example_id, new_program, new_tools_used, reason, update_type, validation_passed, new_execution_result=None):
        """update an example's program solution using a new tool (after abstraction or tool merge)"""
        example = self.get_example(example_id)
        if not example:
            self.logger.warning(f"Example {example_id} not found for update")
            return False
        # 1. store the current state
        old_program = example.program
        old_execution_result = example.execution_result
        old_tools_used = example.tools_used.copy()
        if example.original_program is None:
            example.original_program = old_program
            example.original_execution_result = old_execution_result
        update_record = ExampleUpdateRecord(
            example_id=example_id,
            update_type=update_type,
            old_program=old_program,
            new_program=new_program,
            old_tools_used=old_tools_used,
            new_tools_used=new_tools_used,
            reason=reason,
            validation_passed=validation_passed
        )
        # 2. update fields
        example.program = new_program
        example.tools_used = new_tools_used
        example.last_updated = datetime.now().isoformat()
        if new_execution_result is not None:
            example.execution_result = new_execution_result
        # 3. update abstraction status
        if update_type == 'abstraction':
            if validation_passed.get('execution', False):
                if self.config.require_correctness_for_update:
                    if validation_passed.get('correctness', False): # update abstraction status --> only close for further clustering if program successfully rewritten & validated with new tool
                        example.abstraction_status = "abstracted"
                    else:
                        self.logger.info(f"Example {example_id} failed abstraction CORRECTNESS validation, keeping open for future clustering")
                else:
                    example.abstraction_status = "abstracted"
                # 4. track which tools were abstracted from this example
                for tool in new_tools_used:
                    if tool.startswith('_') and tool not in example.abstracted_tools:
                        example.abstracted_tools.append(tool)
            else:
                self.logger.info(f"Example {example_id} failed abstraction EXECUTION validation, keeping open for future clustering")
        # 5. add to update history
        example.update_history.append({
            'timestamp': update_record.timestamp,
            'update_type': update_type,
            'reason': reason,
            'validation_passed': validation_passed,
            'old_program': old_program,
            'new_program': new_program,
            'old_execution_result': old_execution_result,
            'new_execution_result': new_execution_result
        })
        self.update_records.append(update_record)
        log_example_updated(
            example_id=example_id,
            update_type=update_type,
            reason=reason,
            old_tools=update_record.old_tools_used,
            new_tools=new_tools_used
        )
        return True
    
    def get_examples_for_clustering(self):
        """get examples eligible for clustering (not yet successfully abstracted)"""
        eligible = []
        for ex in self.examples:
            if ex.abstraction_status != "abstracted":
                eligible.append(ex)
        return eligible
    
    def mark_examples_abstracted(self, example_ids, cluster_id):
        """mark examples as part of an abstracted cluster"""
        for ex_id in example_ids:
            example = self.get_example(ex_id)
            if example and cluster_id not in example.abstraction_cluster_ids:
                example.abstraction_cluster_ids.append(cluster_id)
    
    def save_update_history(self, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        history_data = []
        for record in self.update_records:
            history_data.append({
                'example_id': record.example_id,
                'update_type': record.update_type,
                'timestamp': record.timestamp,
                'old_tools_used': record.old_tools_used,
                'new_tools_used': record.new_tools_used,
                'reason': record.reason,
                'validation_passed': record.validation_passed,
                'old_program': record.old_program,
                'new_program': record.new_program
            })
        with open(output_dir / 'example_update_history.json', 'w') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        summary = {
            'total_updates': len(self.update_records),
            'updates_by_type': {},
            'examples_with_updates': len(set(r.example_id for r in self.update_records)),
            'total_examples': len(self.examples)
        }
        for record in self.update_records:
            update_type = record.update_type
            if update_type not in summary['updates_by_type']:
                summary['updates_by_type'][update_type] = 0
            summary['updates_by_type'][update_type] += 1
        with open(output_dir / 'example_update_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def save_embedding_cache(self, cache_path):
        cache_data = {
            'embedding_model': self.config.embedding_model,
            'embeddings': {}
        }
        for question, embedding in self._question_embedding_cache.items():
            cache_data['embeddings'][question] = embedding.tolist()
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, ensure_ascii=False)
        self.logger.info(f"Saved {len(self._question_embedding_cache)} cached embeddings")
    
    def load_embedding_cache(self, cache_path):
        if not cache_path.exists():
            return
        try:
            with open(cache_path) as f:
                cache_data = json.load(f)
            if cache_data.get('embedding_model') != self.config.embedding_model:
                self.logger.warning(f"Embedding cache is for different model, ignoring")
                return
            for question, embedding_list in cache_data.get('embeddings', {}).items():
                self._question_embedding_cache[question] = np.array(embedding_list)
            self.logger.info(f"Loaded {len(self._question_embedding_cache)} cached embeddings")
        except Exception as e:
            self.logger.warning(f"Failed to load embedding cache: {e}")
    
    def load(self, examples):
        """load examples into library (for resuming)"""
        self.examples = []
        self._id_to_example = {}
        self._question_idx_to_example = {}
        for ex in examples:
            self.examples.append(ex)
            self._id_to_example[ex.id] = ex
            self._question_idx_to_example[ex.question.question_index] = ex
        self.logger.info(f"Loaded {len(self.examples)} unique examples into library")

class ToolLibrary:
    """hierarchical library of tools"""
    def __init__(self, config):
        self.config = config
        self.tools = {}  # all tools including deprecated
        self._tools_by_level = {}
        self.logger = get_logger(__name__)
    
    def add_tool(self, tool):
        self.tools[tool.name] = tool
        if tool.level not in self._tools_by_level:
            self._tools_by_level[tool.level] = []
        # add to level list if not already there
        if tool not in self._tools_by_level[tool.level]:
            self._tools_by_level[tool.level].append(tool)
        self.logger.info(f"💎 Added tool {tool.name} at level {tool.level}")
    
    def get_tool(self, name):
        return self.tools.get(name)
    
    def get_all_tools(self):
        """get all non-deprecated tools"""
        return [t for t in self.tools.values() if not t.is_deprecated]
    
    def get_all_tools_including_deprecated(self):
        """get all tools including deprecated ones"""
        return list(self.tools.values())
    
    @property
    def learned_tools(self):
        """get all learned (non-predefined) & non-deprecated tools"""
        return [t for t in self.get_all_tools() if t.level > 0]
    
    def deprecate_tool(self, name, reason):
        """mark a tool as deprecated (but keep it in records)"""
        tool = self.tools.get(name)
        if tool and not tool.is_deprecated:
            tool.is_deprecated = True
            tool.deprecation_reason = reason
            log_tool_deprecation(
                tool_name=name,
                reason=reason,
                metrics={'usage_count': tool.usage_count}
            )
    
    def load(self, tools):
        """load tools into library (including deprecated) (for resuming)"""
        self.tools = {}
        self._tools_by_level = {}
        for tool in tools:
            self.tools[tool.name] = tool
            if tool.level not in self._tools_by_level:
                self._tools_by_level[tool.level] = []
            self._tools_by_level[tool.level].append(tool)
        active_count = len([t for t in tools if not t.is_deprecated])
        deprecated_count = len([t for t in tools if t.is_deprecated])
        self.logger.info(f"Loaded {active_count} active tools and {deprecated_count} deprecated tools")
