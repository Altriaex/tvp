"""
main pipeline orchestrator for TVP.
"""
import json
from tqdm import tqdm
import random
from datetime import datetime

from models import Question, Example, AbstractionInterval, DeduplicationInterval
from state import PipelineState
from logging_utils import *
from agents import *
from library import ExampleLibrary, ToolLibrary
from execution import ExecutionEngine, ExecutionValidator, CorrectnessValidator, ProgramRewriter
from utils import get_predefined_tools, ParallelExecutor, ClusteringEngine, extract_tools_from_program

logger = get_logger(__name__)

class Pipeline:
    """main pipeline orchestrator for TVP library building run"""
    def __init__(self, config):
        self.config = config
        self.state = PipelineState(config)
        self._initialize_components()
        self._initialize_libraries()
        logger.info(f"Pipeline initialized - Run ID: {config.run_id}")
        self._global_cluster_counter = 0
        if self.state.clusters: # initialize global cluster counter from existing state
            max_cluster_num = 0
            for cluster in self.state.clusters:
                if cluster.id.startswith('cluster_'):
                    try:
                        num = int(cluster.id.split('_')[1])
                        max_cluster_num = max(max_cluster_num, num)
                    except:
                        pass
            self._global_cluster_counter = max_cluster_num + 1
    
    def _initialize_components(self):
        """initialize all TVP pipeline components"""
        self.program_generator = ProgramGenerator(self.config)
        self.quality_judge = QualityJudge(self.config)
        self.cluster_analyzer = ClusterAnalyzer(self.config)
        self.tool_abstractor = ToolAbstractor(self.config)
        self.tool_deduplicator = ToolDeduplicator(self.config)
        self.example_library = ExampleLibrary(self.config)
        self.tool_library = ToolLibrary(self.config)
        self.example_library.set_tool_library(self.tool_library)    # connect libraries (for resume)
        self.clustering_engine = ClusteringEngine(self.config)
        self.execution_engine = ExecutionEngine(self.config)
        self.execution_validator = ExecutionValidator(self.config)
        self.correctness_validator = CorrectnessValidator(self.config)
        self.rewriter = ProgramRewriter(self.config)
        self.parallel_executor = ParallelExecutor(self.config)
    
    def _initialize_libraries(self):
        """load existing libraries (resume) or initialize new ones"""
        if self.config.resume_run_id:
            checkpoint_path = self.state.state_dir / "checkpoint.json"
            if checkpoint_path.exists():
                logger.info("Loading existing state...")
                self.state.load()
                self.example_library.load(self.state.examples)
                self.tool_library.load(self.state.tools)
                active_tools = self.tool_library.get_all_tools()
                all_tools = self.tool_library.get_all_tools_including_deprecated()
                deprecated_count = len(all_tools) - len(active_tools)
                logger.info(f"Loaded {len(active_tools)} active tools and {deprecated_count} deprecated tools")
                logger.info(f"Total tools in library (including deprecated): {len(all_tools)}")
                if deprecated_count > 0:
                    deprecated_names = [t.name for t in all_tools if t.is_deprecated]
                    logger.debug(f"Deprecated tool names: {deprecated_names}")  # verify deprecated tool names for debugging
                self.correctness_validator.set_tool_library(self.tool_library)
                cache_path = self.state.state_dir / "embedding_cache.json" 
                self.example_library.load_embedding_cache(cache_path)
                return
            else:
                logger.warning(f"No existing state found for run_id: {self.config.resume_run_id}")
                logger.info("Initializing new state instead...")
        # initialize new state with predefined tools
        logger.info("Initializing new libraries...")
        predefined_tools = get_predefined_tools(self.config.dataset)
        logger.info(f"Loading {len(predefined_tools)} predefined tools for dataset: {self.config.dataset}")
        for tool in predefined_tools:
            self.tool_library.add_tool(tool)
        self.state.tools = self.tool_library.get_all_tools_including_deprecated()
        logger.info(f"Initialized with {len(self.tool_library.get_all_tools())} tools")
        self.correctness_validator.set_tool_library(self.tool_library)
    
    def run(self):
        logger.info("🚀 Starting pipeline")
        start_time = datetime.now()
        log_resume_info(self.state.current_iteration, self.state.questions_processed)
        for iteration in range(self.state.current_iteration, self.config.num_iterations + 1):
            self._run_iteration(iteration)
        cache_path = self.state.state_dir / "embedding_cache.json"
        self.example_library.save_embedding_cache(cache_path)
        log_final_summary(
            self.state.questions_processed,
            len(self.state.examples),
            len(self.tool_library.get_all_tools()),
            datetime.now() - start_time
        )
    
    def _run_iteration(self, iteration):
        logger.info(f"🔄 Starting iteration {iteration}")
        log_iteration_start(iteration, self.config.num_iterations)
        self.state.current_iteration = iteration
        questions = self._load_questions(iteration)
        # track already processed questions (resuming)
        iteration_tracker_path = self.state.state_dir / f"iteration_{iteration}_tracker.json"
        processed_in_this_iteration = set()
        iteration_questions_processed = 0
        if iteration_tracker_path.exists():
            with open(iteration_tracker_path) as f:
                tracker_data = json.load(f)
                processed_in_this_iteration = set(tracker_data.get('processed_indices', []))
                iteration_questions_processed = tracker_data.get('count', 0)
                logger.info(f"Resuming iteration {iteration} - Already processed {iteration_questions_processed} questions")
        questions_to_process = [q for q in questions if q.question_index not in processed_in_this_iteration]
        pbar = tqdm(questions_to_process, 
                desc=f"Iteration {iteration} - Processing questions",
                initial=iteration_questions_processed + 1,
                total=len(questions))
        questions_in_iteration_start = self.state.questions_processed
        for question in questions:
            if question.question_index in processed_in_this_iteration:
                logger.info(f"Skipping already processed question {question.question_index}")
                continue
            log_question_processing(question.question_index, question.question)
            example = self._process_question(question, iteration) 
            processed_in_this_iteration.add(question.question_index)
            iteration_questions_processed += 1
            self.state.questions_processed += 1
            if example: # passed example quality threshold
                self.example_library.add_example(example)   # add to example lib
                log_question_complete(question.question_index, True, example.quality_rating)
            else:
                log_question_complete(question.question_index, False)
            self.state.examples = self.example_library.examples.copy()
            # check for abstraction interval after processing the question
            questions_since_iteration_start = self.state.questions_processed - questions_in_iteration_start
            if (self.config.abstraction_interval > 0 and 
                questions_since_iteration_start > 0 and
                questions_since_iteration_start % self.config.abstraction_interval == 0):
                logger.info(f"🎯 Abstraction interval triggered after processing question {question.question_index}")
                interval_number = questions_since_iteration_start // self.config.abstraction_interval
                self._abstract_and_deduplicate(iteration, interval_num=interval_number)
            # check for deduplication interval after processing the question
            if (self.config.enable_deduplication and 
                self.config.deduplication_interval > 0 and
                questions_since_iteration_start > 0 and
                questions_since_iteration_start % self.config.deduplication_interval == 0):
                logger.info(f"🎯 Deduplication interval triggered after processing question {question.question_index}")
                interval_number = questions_since_iteration_start // self.config.deduplication_interval
                self._perform_deduplication(iteration, interval_num=interval_number)
            if pbar:
                pbar.update(1)
            # save iteration tracker of processed question (incl. abstraction/deduplication)
            iteration_tracker_path.parent.mkdir(parents=True, exist_ok=True)
            with open(iteration_tracker_path, 'w') as f:
                json.dump({
                    'iteration': iteration,
                    'count': iteration_questions_processed,
                    'processed_indices': list(processed_in_this_iteration)
                }, f, ensure_ascii=False)
            self.state.save()
            # save embedding cache
            cache_path = self.state.state_dir / "embedding_cache.json"
            self.example_library.save_embedding_cache(cache_path)
        if pbar:
            pbar.close()
        # clean up iteration tracker after entire iter is complete
        if iteration_questions_processed == len(questions):
            if iteration_tracker_path.exists():
                iteration_tracker_path.unlink()
                logger.info(f"Removed completed iteration tracker for iteration {iteration}")
        # save state after iter completion
        self.state.save()
        logger.info(f"✅ Completed iteration {iteration}")
    
    def _load_questions(self, iteration):
        sampled_questions_path = self.config.results_dir / "sampled_questions.json" # check if existing sampled questions exist (resuming)
        if sampled_questions_path.exists():
            logger.info(f"Loading existing sampled questions from {sampled_questions_path}")
            with open(sampled_questions_path) as f:
                questions_data = json.load(f)
            questions = []
            for q_data in questions_data:
                questions.append(Question(**q_data))
            if self.config.reorder_questions_per_iteration:
                random.seed(self.config.random_seed + iteration)
                questions = random.sample(questions, len(questions))
                logger.info(f"Applied random ordering for iteration {iteration}")
            logger.info(f"Loaded {len(questions)} pre-sampled questions")
            return questions
        logger.info("No sampled questions found, loading from annotations") # no sampled questions exist - load from dataset source annotations
        with open(self.config.annotations_path) as f:
            all_questions_data = json.load(f)
        if isinstance(all_questions_data, list):
            questions_list = all_questions_data
        elif isinstance(all_questions_data, dict) and 'questions' in all_questions_data:
            questions_list = all_questions_data['questions']
        all_questions = []
        for q_data in questions_list:
            all_questions.append(Question(**q_data))
        if self.config.reorder_questions_per_iteration:
            random.seed(self.config.random_seed + iteration)
            sorted_questions = random.sample(all_questions, len(all_questions))
        else:
            sorted_questions = all_questions
        num_to_process = self.config.num_questions if self.config.num_questions > 0 else len(sorted_questions)
        selected_questions = sorted_questions[:num_to_process]
        sampled_questions_path = self.config.results_dir / "sampled_questions.json" # save sampled questions for resume
        with open(sampled_questions_path, 'w') as f:
            json.dump([q.__dict__ for q in selected_questions], f, indent=2, ensure_ascii=False)
        return selected_questions

    def _process_question(self, question, iteration):
        question_dir = (self.config.results_dir / f"iterations/iteration_{iteration}/questions/q_{question.question_index}")
        question_dir.mkdir(parents=True, exist_ok=True)
        # 1. retrieve similar examples
        retrieved_examples, similarities = self._retrieve_examples(question, question_dir)
        # 2. generate program candidates (parallel)
        candidates = self._generate_candidates(question, retrieved_examples, question_dir)  # filtered out error during generation
        # 3. execute programs (sequential)
        executed_candidates = self._execute_candidates(candidates, question_dir)
        # 4. filter successful executions - (none answers = exe failure)
        successful_candidates = [c for c in executed_candidates if not c.execution_error and c.answer is not None]
        log_execution_summary(len(candidates), len(successful_candidates))
        if not successful_candidates:
            logger.warning(f"No successful executions for question {question.question_index}")  # error during execution
            self._save_best_program(question_dir, executed_candidates, None)    # save dummy best program info even if all execution failed
            return None
        # 5. judge quality (parallel)
        judged_candidates = self._judge_candidates(successful_candidates, executed_candidates, question_dir)
        if not judged_candidates:
            logger.warning(f"No candidates could be judged for question {question.question_index}")  # error during judge
            self._save_best_program(question_dir, executed_candidates, None)    # save dummy best program info even if all judge failed
            return None
        # 6. select best candidate
        best_candidate = max(judged_candidates, key=lambda c: c.quality_rating)
        # 7. save best program info
        self._save_best_program(question_dir, executed_candidates, best_candidate)
        logger.info(f"🏆 Selected best program for question {question.question_index}:")
        logger.info(f"  - Candidate index: {best_candidate.candidate_index}")
        logger.info(f"  - Quality rating: {best_candidate.quality_rating}")
        logger.info(f"  - Answer: {best_candidate.answer}")
        logger.info(f"  - Tools used: {extract_tools_from_program(best_candidate.code)}")
        if best_candidate.quality_rating < self.config.quality_threshold:
            logger.info(f"Best program quality {best_candidate.quality_rating} below threshold")
            return None
        # 8. create potential example out of best candidate (not yet checked for example lib entry)
        example = self._create_example(best_candidate, iteration)
        return example
    
    def _retrieve_examples(self, question, output_dir):
        retrieval_dir = output_dir / "retrieval"
        retrieval_dir.mkdir(exist_ok=True)
        retrieved, similarities = self.example_library.retrieve_similar(
            question.question,
            max_examples=self.config.max_retrieved_examples,
            threshold=self.config.similarity_threshold,
            exclude_question_index=question.question_index  # exclude current question from similarity-based retrieval
        )
        retrieval_data = {
            'query': question.question,
            'query_question_index': question.question_index,
            'query_embedding_computed': True,
            'total_examples_in_library': len(self.example_library.examples),
            'retrieved_count': len(retrieved),
            'similarities': similarities,
            'retrieved_examples': [ex.to_dict() for ex in retrieved]
        }
        with open(retrieval_dir / "retrieval_details.json", 'w') as f:
            json.dump(retrieval_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Retrieved {len(retrieved)} similar examples (excluding question itself: {question.question_index})")
        return retrieved, similarities

    def _generate_candidates(self, question, retrieved_examples, output_dir):
        generation_dir = output_dir / "generation"
        generation_dir.mkdir(exist_ok=True)
        all_tools = self.tool_library.get_all_tools()
        logger.debug(f"Generating with {len(all_tools)} tools available")
        tasks = []
        for i in range(self.config.programs_per_question):
            tasks.append({
                'question': question,
                'examples': retrieved_examples,
                'tools': all_tools,
                'index': i,
                'output_dir': generation_dir
            })
        candidates = self.parallel_executor.execute_parallel(
            self.program_generator.generate_single,
            tasks
        )
        candidates = [c for c in candidates if c is not None]   # filter out none candidate (error in generation)
        return candidates
    
    def _execute_candidates(self, candidates, output_dir):
        execution_dir = output_dir / "execution"
        execution_dir.mkdir(exist_ok=True)
        all_tools = self.tool_library.get_all_tools()
        for candidate in candidates:
            logger.info(f"🔧 Program execution")
            exec_result = self.execution_engine.execute(
                candidate,
                all_tools,
                execution_dir / f"candidate_{candidate.candidate_index}"
            )
            candidate.execution_result = exec_result['namespace']
            candidate.execution_error = exec_result['error']
            candidate.execution_details = exec_result.get('error_details')
            candidate.answer = exec_result['answer']
            if not exec_result['error']:
                logger.info(f"  - Execution successful, answer: {exec_result['answer']}")
            else:
                logger.info(f"  - Execution failed: {exec_result['error']}")
        return candidates
    
    def _judge_candidates(self, successful_candidates, all_candidates, output_dir):
        judging_dir = output_dir / "judging"
        judging_dir.mkdir(exist_ok=True)
        all_tools = self.tool_library.get_all_tools()
        tasks = []
        for candidate in successful_candidates:
            tasks.append({
                'candidate': candidate,
                'output_dir': judging_dir / f"candidate_{candidate.candidate_index}",
                'tools': all_tools
            })
        judged_candidates = self.parallel_executor.execute_parallel(
            self.quality_judge.judge_single,
            tasks
        )
        judged_candidates = [c for c in judged_candidates if c is not None]
        if not judged_candidates:
            logger.warning(f"No candidates were successfully judged for question {successful_candidates[0].question.question_index}")   # error during judge
            return []
        scores_data = {
            'question_index': successful_candidates[0].question.question_index,
            'scores': [],
            'best_score': max(c.quality_rating for c in judged_candidates) if judged_candidates else 0
        }
        judged_dict = {c.candidate_index: c for c in judged_candidates}
        for original_candidate in all_candidates:
            if original_candidate.candidate_index in judged_dict:
                judged = judged_dict[original_candidate.candidate_index]
                scores_data['scores'].append({
                    'candidate_index': original_candidate.candidate_index,
                    'quality_rating': judged.quality_rating,
                    'quality_reasoning': judged.quality_reasoning,
                    'execution_success': True
                })
            else:
                scores_data['scores'].append({
                    'candidate_index': original_candidate.candidate_index,
                    'quality_rating': None,
                    'quality_reasoning': 'Not judged due to execution failure or None result',
                    'execution_success': False,
                    'execution_error': original_candidate.execution_error
                })
        with open(judging_dir / "scores.json", 'w') as f:
            json.dump(scores_data, f, indent=2, ensure_ascii=False)
        return judged_candidates
    
    def _create_example(self, candidate, iteration):
        example_id = f"q{candidate.question.question_index}"
        if candidate.question.question in self.example_library._question_embedding_cache:
            question_embedding = self.example_library._question_embedding_cache[candidate.question.question]
            logger.debug(f"Using cached embedding for question: {candidate.question.question}...")
        else:
            question_embedding = self.example_library.embedding_manager.embed_text(candidate.question.question)
            self.example_library._cache_question_embedding(candidate.question.question, question_embedding)
        example = Example(
            id=example_id,
            question=candidate.question,
            program=candidate.code,
            quality_rating=candidate.quality_rating,
            execution_result=candidate.execution_result,  # store full execution namespace
            question_embedding=question_embedding,
            tools_used=extract_tools_from_program(candidate.code),
            iteration=iteration
        )
        return example
    
    def _save_best_program(self, question_dir, all_candidates, best_candidate):
        best_program_data = {
            'question_index': all_candidates[0].question.question_index if all_candidates else None,
            'question': all_candidates[0].question.question if all_candidates else None,
            'total_candidates': len(all_candidates),
            'successful_executions': len([c for c in all_candidates if not c.execution_error and c.answer is not None]),
            'timestamp': datetime.now().isoformat()
        }
        if best_candidate:
            best_program_data.update({
                'selected': True,
                'candidate_index': best_candidate.candidate_index,
                'quality_rating': best_candidate.quality_rating,
                'quality_reasoning': best_candidate.quality_reasoning,
                'program': best_candidate.code,
                'execution_result': best_candidate.answer,
                'answer': best_candidate.answer,
                'tools_used': extract_tools_from_program(best_candidate.code),
                'added_to_library': best_candidate.quality_rating >= self.config.quality_threshold
            })
        else:
            best_program_data.update({
                'selected': False,
                'reason': 'No successful candidates' if all_candidates else 'No candidates generated',
                'added_to_library': False
            })
            if all_candidates:  # try to find best among any judged candidates
                judged = [c for c in all_candidates if c.quality_rating is not None]
                if judged:
                    best_failed = max(judged, key=lambda c: c.quality_rating)
                    best_program_data['best_failed_attempt'] = {
                        'candidate_index': best_failed.candidate_index,
                        'quality_rating': best_failed.quality_rating,
                        'quality_reasoning': best_failed.quality_reasoning,
                        'execution_error': best_failed.execution_error,
                        'answer': best_failed.answer
                    }
                else:   # no judged candidates, pick first
                    for c in all_candidates:
                        if c.execution_error or c.answer is None:
                            best_program_data['best_failed_attempt'] = {
                                'candidate_index': c.candidate_index,
                                'quality_rating': None,
                                'quality_reasoning': 'Not judged due to execution failure or None result',
                                'execution_error': c.execution_error if c.execution_error else 'None result',
                                'answer': c.answer
                            }
                            break
        with open(question_dir / "best_program.json", 'w') as f:
            json.dump(best_program_data, f, indent=2, ensure_ascii=False)
    
    def _abstract_and_deduplicate(self, iteration, interval_num=None):
        global_interval_number = len(self.state.abstraction_intervals) + 1
        abstraction_dir = self.config.results_dir / f"abstraction/iteration_{iteration}_interval_{global_interval_number}"
        abstraction_dir.mkdir(parents=True, exist_ok=True)
        interval = AbstractionInterval(
            interval_number=global_interval_number,
            start_question_index=self.state.questions_processed - self.config.abstraction_interval + 1 if interval_num else self.state.questions_processed,
            end_question_index=self.state.questions_processed,
            started_at=datetime.now().isoformat()
        )
        logger.info(f"🔮 Starting Abstraction Interval #{global_interval_number}")
        log_abstraction_interval(
            self.state.questions_processed,
            len(self.example_library.examples),
            len(self.tool_library.learned_tools)
        )
        # 1. cluster eligible examples (initially: embedding-similarity; then: llm analysis)
        clusters, final_clustering_log = self._cluster_examples(abstraction_dir)
        interval.clusters_found = len(clusters)
        if not clusters:
            logger.info(f"No abstractable clusters found for interval #{global_interval_number}")
            interval.completed_at = datetime.now().isoformat()
            self.state.abstraction_intervals.append(interval)
            self.state.abstraction_intervals_completed = len(self.state.abstraction_intervals)
            self.state.save()
            return
        log_clustering_result(global_interval_number,
            final_clustering_log['initial_groups'],
            final_clustering_log['clusters_found']
        )
        # 2. abstract tools
        tools_before = len(self.tool_library.learned_tools)
        self._abstract_tools(clusters, abstraction_dir, global_interval_number)
        tools_after = len(self.tool_library.learned_tools)
        interval.tools_created = tools_after - tools_before
        interval.completed_at = datetime.now().isoformat()
        self.state.abstraction_intervals.append(interval)
        self.state.abstraction_intervals_completed = len(self.state.abstraction_intervals)
        self.state.save()
        logger.info(f"✅ Completed Abstraction Interval #{global_interval_number} - Created {interval.tools_created} tools from {interval.clusters_found} clusters")
    
    def _cluster_examples(self, output_dir):
        clustering_dir = output_dir / "clustering"
        clustering_dir.mkdir(parents=True, exist_ok=True)
        current_examples = self.example_library.get_examples_for_clustering()   # get only eligible examples (not yet abstracted)
        if len(current_examples) < self.config.min_cluster_size:
            logger.info(f"Not enough eligible examples for clustering: {len(current_examples)} < {self.config.min_cluster_size}")
            all_clusters = []
            final_clustering_log = {
                'initial_groups': 0,
                'groups_analyzed': 0,
                'clusters_found': 0,
                'clusters': []
            }
            return all_clusters, final_clustering_log
        clustering_log_path = clustering_dir / "clustering_log.json"
        clustering_log = {
            'timestamp': datetime.now().isoformat(),
            'total_examples': len(self.example_library.examples),
            'eligible_examples': len(current_examples),
            'threshold': self.config.clustering_threshold,
            'min_cluster_size': self.config.min_cluster_size
        }
        logger.info(f"Clustering {len(current_examples)} eligible examples (out of {len(self.example_library.examples)} total)")
        # 1. group by embedding similarity
        initial_groups, clustering_info = self.clustering_engine.group_by_similarity(
            current_examples,
            threshold=self.config.clustering_threshold,
            return_detailed_info=True
        )
        clustering_log.update({
            'groups_found': len(initial_groups),
            'clustering_info': clustering_info
        })
        with open(clustering_log_path, 'w') as f:
            json.dump(clustering_log, f, indent=2, ensure_ascii=False)
        all_clusters = []
        for group_idx, group in enumerate(initial_groups):
            if len(group) < self.config.min_cluster_size:
                logger.info(f"Skipping group {group_idx} with only {len(group)} examples (min: {self.config.min_cluster_size})")
                continue
            # 2. if embedding-based group meets min_cluster_size --> llm further analyze group
            clusters = self.cluster_analyzer.analyze_group(
                group,
                clustering_dir / f"group_{group_idx}"
            )
            for cluster in clusters:
                cluster.id = f"cluster_{self._global_cluster_counter}"
                self._global_cluster_counter += 1
            all_clusters.extend(clusters)
        for cluster in all_clusters:
            self.state.clusters.append(cluster)
        final_clustering_log = {
            'initial_groups': len(initial_groups),
            'groups_analyzed': len([g for g in initial_groups if len(g) >= self.config.min_cluster_size]),
            'clusters_found': len(all_clusters),
            'clusters': [c.to_dict() for c in all_clusters]
        }
        with open(clustering_dir / "final_clusters.json", 'w') as f:
            json.dump(final_clustering_log, f, indent=2, ensure_ascii=False)
        return all_clusters, final_clustering_log
    
    def _abstract_tools(self, clusters, output_dir, interval_number):
        tools_dir = output_dir / "tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        # 1. check for min_cluster_size & abstraction_potential (after both embedding-similarity grouping & llm analysis) 
        abstractable_clusters = [
            c for c in clusters 
            if not c.is_abstracted and
            len(c.example_ids) >= self.config.min_cluster_size and 
            c.abstraction_potential >= self.config.abstraction_potential_threshold
        ]
        if not abstractable_clusters:
            logger.info(f"No clusters meet abstraction threshold for interval #{interval_number}")
            return
        logger.info(f"🔧 Abstracting tools from {len(abstractable_clusters)} clusters in interval #{interval_number}")
        log_abstraction_start(len(abstractable_clusters))
        abstraction_summary = {
            'timestamp': datetime.now().isoformat(),
            'interval_number': interval_number,
            'clusters_to_abstract': len(abstractable_clusters),
            'tools_created': [],
            'failed_abstractions': []
        }
        for cluster in abstractable_clusters:
            logger.info(f"Processing cluster {cluster.id}")
            # 2. abstract tools, rewrite programs, validate
            tool, validation_result = self._abstract_cluster(cluster, output_dir)
            if tool and validation_result and validation_result.passed: # abstraction successful
                self.tool_library.add_tool(tool)
                self.state.tools = self.tool_library.get_all_tools_including_deprecated()
                log_tool_abstracted(
                    tool.name,
                    tool.level,
                    len(cluster.example_ids),
                    validation_result.execution_success_rate
                )
                # 3. update abstracted examples in example lib using new tool
                updated_count = self._update_examples_after_abstraction(
                    tool, validation_result, tools_dir / f"cluster_{cluster.id}"
                )
                successfully_updated_ids = []
                for rewrite in validation_result.successful_rewrites:
                    if rewrite.get('validation_passed', {}).get('execution', False):
                        if not self.config.require_correctness_for_update or rewrite.get('validation_passed', {}).get('correctness', False):
                            successfully_updated_ids.append(rewrite['example_id'])
                if successfully_updated_ids:
                    self.example_library.mark_examples_abstracted(successfully_updated_ids, cluster.id)
                if updated_count > 0:
                    log_example_updates_summary(
                        updated_count,
                        len(cluster.example_ids),
                        'abstraction'
                    )
                abstraction_summary['tools_created'].append({
                    'tool_name': tool.name,
                    'cluster_id': cluster.id,
                    'examples_updated': updated_count,
                    'validation_rates': {
                        'execution': validation_result.execution_success_rate,
                        'correctness': validation_result.correctness_rate
                    }
                })
            else:   # abstraction failed
                abstraction_summary['failed_abstractions'].append({
                    'cluster_id': cluster.id,
                    'reason': 'Validation failed' if validation_result else 'Tool generation failed'
                })
        with open(tools_dir / "abstraction_summary.json", 'w') as f:
            json.dump(abstraction_summary, f, indent=2, ensure_ascii=False)
        updates_dir = output_dir / "example_updates"
        self.example_library.save_update_history(updates_dir)
        self.state.save()
    
    def _abstract_cluster(self, cluster, abstraction_dir):
        cluster_dir = abstraction_dir / f"tools/cluster_{cluster.id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        examples = [self.example_library.get_example(ex_id) for ex_id in cluster.example_ids]
        logger.info(f"Abstracting cluster {cluster.id} with {len(examples)} examples")
        all_attempts = []
        for retry in range(self.config.max_abstraction_retries):
            logger.info(f"Abstraction attempt {retry + 1}/{self.config.max_abstraction_retries}")
            # 1. create tool abstraction
            tool = self.tool_abstractor.abstract_from_cluster(
                cluster,
                examples,
                self.tool_library.get_all_tools(),  # active tools for implementation
                self.tool_library.get_all_tools_including_deprecated(),  # all created tools for forbidden names
                cluster_dir / f"attempt_{retry}"
            )
            if not tool:
                logger.warning("Failed to generate tool abstraction")
                all_attempts.append({
                    'retry': retry,
                    'tool': None,
                    'validation_result': None,
                    'failure_reason': 'Tool generation failed'
                })
                continue
            # 2. validate tool
            validation_result = self._validate_tool(tool, examples, cluster_dir / f"validation_{retry}")
            if tool:
                attempt_info = {
                    'retry': retry,
                    'tool': {
                        'name': tool.name,
                        'signature': tool.signature,
                        'docstring': tool.docstring,
                        'implementation': tool.implementation
                    },
                    'validation_result': {
                        'execution_success_rate': validation_result.execution_success_rate,
                        'execution_successes': validation_result.execution_successes,
                        'execution_failures': validation_result.execution_failures,
                        'correctness_rate': validation_result.correctness_rate,
                        'correct_results': validation_result.correct_results,
                        'correctness_checks': validation_result.correctness_checks,
                        'total_examples': validation_result.total_examples,
                        'execution_errors': [],
                        'correctness_failures': []
                    } if validation_result else None
                }
            else:   # error in tool parsing - store minimal info
                attempt_info = {
                    'retry': retry,
                    'tool': None,
                    'validation_result': None,
                    'failure_reason': 'Tool parsing failed'
                }
            all_attempts.append(attempt_info)
            if validation_result.passed:
                logger.info(f"🎉 Tool {tool.name} passed validation on attempt {retry + 1}")
                tool.source_cluster_id = cluster.id
                cluster.is_abstracted = True
                cluster.abstracted_tool_names.append(tool.name)
                with open(cluster_dir / "final_status.json", 'w') as f:
                    json.dump({
                        'success': True,
                        'tool_name': tool.name,
                        'retry_count': retry,
                        'execution_success_rate': validation_result.execution_success_rate,
                        'correctness_rate': validation_result.correctness_rate,
                        'all_attempts': all_attempts
                    }, f, indent=2, ensure_ascii=False)
                return tool, validation_result
            logger.warning(f"Validation failed - Execution: {validation_result.execution_success_rate:.2%}, "
                         f"Correctness: {validation_result.correctness_rate:.2%}")
        # all attempts failed
        with open(cluster_dir / "final_status.json", 'w') as f: 
            json.dump({
                'success': False,
                'retry_count': self.config.max_abstraction_retries,
                'reason': 'Validation failed after all retries',
                'all_attempts': all_attempts
            }, f, indent=2, ensure_ascii=False)
        return None, None
    
    def _validate_tool(self, tool, examples, validation_dir):
        validation_dir.mkdir(parents=True, exist_ok=True)
        self.correctness_validator.set_current_tool(tool)
        all_tools = self.tool_library.get_all_tools()
        # 1. validate execution success
        stage1_result = self.execution_validator.validate(
            tool, examples, all_tools, validation_dir / "stage1_execution"
        )
        logger.info(f"📊 Stage 1: Execution validation - Success rate: {stage1_result.execution_success_rate:.2%}")
        logger.info(f"Validation details for tool {tool.name}:")
        logger.info(f"  - Total examples: {stage1_result.total_examples}")
        logger.info(f"  - Execution successes: {stage1_result.execution_successes}")
        logger.info(f"  - Execution failures: {stage1_result.execution_failures}")
        if stage1_result.execution_success_rate < self.config.min_execution_success_rate:
            logger.warning(f"Tool {tool.name} failed execution validation")
            stage1_result.passed = False
            return stage1_result
        # 2. validate correctness of divergent results
        stage2_result = self.correctness_validator.validate(
            stage1_result, validation_dir / "stage2_correctness"
        )
        logger.info(f"📊 Stage 2: Correctness validation - Overall success rate (all examples, including non-divergent results): {stage2_result.overall_correctness_rate:.2%}")
        if stage2_result.correctness_details:
            logger.info(f"  - Divergent results checked: {len(stage2_result.correctness_details)}")
            correct_count = len([d for d in stage2_result.correctness_details if d.get('verdict') == 'CORRECT'])
            logger.info(f"  - Correct divergent results: {correct_count}")
            for i, detail in enumerate(stage2_result.correctness_details):
                logger.info(f"  - Example {i+1}:")
                logger.info(f"    - Original answer: {detail['original_answer']}")
                logger.info(f"    - New answer: {detail['new_answer']}")
                logger.info(f"    - Verdict: {detail.get('verdict', 'unknown')}")
        if stage2_result.overall_correctness_rate < self.config.min_correctness_rate:
            logger.warning(f"Tool {tool.name} failed correctness validation")
            stage2_result.passed = False
            return stage2_result
        stage2_result.passed = True
        with open(validation_dir / "validation_summary.json", 'w') as f:
            json.dump({
                'tool_name': tool.name,
                'passed': stage2_result.passed,
                'execution': {
                    'success_rate': stage1_result.execution_success_rate,
                    'successes': stage1_result.execution_successes,
                    'failures': stage1_result.execution_failures
                },
                'correctness': {
                    'success_rate': stage2_result.overall_correctness_rate,
                    'checks': stage2_result.correctness_checks,
                    'correct': stage2_result.correct_results
                }
            }, f, indent=2, ensure_ascii=False)
        return stage2_result
    
    def _update_examples_after_abstraction(self, tool, validation_result, abstraction_dir):
        if not validation_result.successful_rewrites:
            return 0
        rewrites_to_apply = []
        for rewrite in validation_result.successful_rewrites:
            if not rewrite['validation_passed'].get('execution', False):
                continue
            if self.config.require_correctness_for_update:  # only update examples that passed validation if divergent results
                if not rewrite['validation_passed'].get('correctness', False):
                    continue
            rewrites_to_apply.append(rewrite)
        if not rewrites_to_apply:
            logger.info(f"No rewrites passed validation criteria for tool {tool.name}")
            return 0
        updated_count = self.example_library.update_examples_for_new_tool(
            tool_name=tool.name,
            successful_rewrites=rewrites_to_apply,
            update_type='abstraction'
        )
        updates_dir = abstraction_dir / "example_updates"
        self.example_library.save_update_history(updates_dir)
        self.state.examples = self.example_library.examples.copy()
        return updated_count
    
    def _perform_deduplication(self, iteration, interval_num=None):
        global_interval_number = len(self.state.deduplication_intervals) + 1
        dedup_dir = self.config.results_dir / f"deduplication/iteration_{iteration}_interval_{global_interval_number}"
        dedup_dir.mkdir(parents=True, exist_ok=True)
        interval = DeduplicationInterval(
            interval_number=global_interval_number,
            start_question_index=self.state.questions_processed - self.config.deduplication_interval + 1 if interval_num else self.state.questions_processed,
            end_question_index=self.state.questions_processed,
            started_at=datetime.now().isoformat()
        )
        logger.info(f"🔀 Starting Deduplication Interval #{global_interval_number}")
        log_deduplication_start(len(self.tool_library.learned_tools))
        learned_tools = self.tool_library.learned_tools
        interval.tools_before = len(learned_tools)
        dedup_summary = {
            'timestamp': datetime.now().isoformat(),
            'interval_number': global_interval_number,
            'tools_before': len(learned_tools),
            'eligible_tools': 0,
            'merge_groups': 0,
            'merges': [],
            'reason_skipped': None
        }
        if len(learned_tools) < 2:
            logger.info("Not enough learned tools for deduplication")
            dedup_summary['reason_skipped'] = f"Not enough learned tools ({len(learned_tools)} < 2)"
            interval.tools_after = interval.tools_before
            interval.completed_at = datetime.now().isoformat()
            self._track_deduplication_interval(interval, dedup_summary, dedup_dir)
            return
        # filter tools by usage count criteria
        eligible_tools = [t for t in learned_tools if t.usage_count >= self.config.min_tool_uses and t.usage_count <= self.config.max_tool_uses]
        dedup_summary['eligible_tools'] = len(eligible_tools)
        interval.eligible_tools = len(eligible_tools)
        if len(eligible_tools) < 2:
            logger.info(f"Not enough tools meet usage requirement (min uses: {self.config.min_tool_uses}, max uses: {self.config.max_tool_uses})")
            dedup_summary['reason_skipped'] = f"Not enough eligible tools ({len(eligible_tools)} < 2) with min uses >= {self.config.min_tool_uses} & max uses <= {self.config.max_tool_uses}"
            dedup_summary['tool_usage_stats'] = {
                t.name: t.usage_count for t in learned_tools
            }
            interval.tools_after = interval.tools_before
            interval.completed_at = datetime.now().isoformat()
            self._track_deduplication_interval(interval, dedup_summary, dedup_dir)
            return
        # 1. find potentially duplicate tools (grouping similar tools)
        merge_groups = self.tool_deduplicator.find_duplicates(
            eligible_tools,
            dedup_dir / "duplicate_analysis"
        )
        dedup_summary['merge_groups'] = len(merge_groups)
        interval.merge_groups_found = len(merge_groups)
        if not merge_groups:
            logger.info("No duplicate tools found")
            dedup_summary['reason_skipped'] = "No duplicate tools found above similarity threshold"
            interval.tools_after = interval.tools_before
            interval.completed_at = datetime.now().isoformat()
            self._track_deduplication_interval(interval, dedup_summary, dedup_dir)
            return
        log_duplicates_found(len(merge_groups))
        for i, group in enumerate(merge_groups):
            logger.info(f"🔧 Processing merge group {i+1}/{len(merge_groups)}")
            tools_to_merge = [self.tool_library.get_tool(name) for name in group['tool_names']]
            tools_to_merge = [t for t in tools_to_merge if t is not None]
            if len(tools_to_merge) < 2:
                continue
            # 2. merge duplicate tools
            merged_tool, merge_validation_result = self._merge_tools_with_retry(
                tools_to_merge,
                group['merge_strategy'],
                group['reasoning'],
                dedup_dir / f"merge_{i}"
            )
            if merged_tool and merge_validation_result and merge_validation_result.passed:
                for tool in tools_to_merge:
                    self.tool_library.deprecate_tool(tool.name, f"Merged into {merged_tool.name}")  # update tool library (deprecate duplicates)
                self.tool_library.add_tool(merged_tool)
                self.state.tools = self.tool_library.get_all_tools_including_deprecated()
                updated_count = 0
                if self.config.update_examples_after_deduplication: # update examples with validated program rewrites using merged tool
                    updated_count = self._update_examples_after_merge(
                        merged_tool,
                        merge_validation_result,
                        dedup_dir / f"merge_{i}"
                    )
                log_tool_merged(
                    [t.name for t in tools_to_merge],
                    merged_tool.name,
                    updated_count
                )
                dedup_summary['merges'].append({
                    'original_tools': [t.name for t in tools_to_merge],
                    'merged_tool': merged_tool.name,
                    'examples_updated': updated_count,
                    'merge_strategy': group['merge_strategy']
                })
            else:
                logger.warning(f"Failed to merge tools: {[t.name for t in tools_to_merge]}")
        interval.tools_after = len(self.tool_library.learned_tools)
        interval.completed_at = datetime.now().isoformat()
        dedup_summary['tools_after'] = interval.tools_after
        self._track_deduplication_interval(interval, dedup_summary, dedup_dir)
        if dedup_summary['merges']:
            log_example_updates_summary(
                sum(m['examples_updated'] for m in dedup_summary['merges']),
                len(self.example_library.examples),
                'deduplication'
            )
        logger.info(f"✅ Completed Deduplication Interval #{global_interval_number} - Consolidated {len(merge_groups)} groups of similar tools")
    
    def _track_deduplication_interval(self, interval, dedup_summary, dedup_dir):
        with open(dedup_dir / "deduplication_summary.json", 'w') as f:
            json.dump(dedup_summary, f, indent=2, ensure_ascii=False)
        self.state.deduplication_intervals.append(interval)
        self.state.deduplication_rounds = len(self.state.deduplication_intervals)
        self.state.save()

    def _merge_tools_with_retry(self, tools_to_merge, merge_strategy, merge_strategy_reason, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        all_attempts = []
        # 1. find examples using to-be-merged tools
        examples_to_update = []
        for example in self.example_library.examples:
            tools_used = extract_tools_from_program(example.program)
            for tool in tools_to_merge:
                if tool.name in tools_used:
                    examples_to_update.append(example)
                    break
        logger.info(f"Found {len(examples_to_update)} examples using tools to be merged")
        for retry in range(self.config.max_merge_retries):
            logger.info(f"🖇️ Merge attempt {retry + 1}/{self.config.max_merge_retries}")
            # 2. merge
            merged_tool = self.tool_deduplicator.merge_tools(
                    tools_to_merge,
                    merge_strategy,
                    merge_strategy_reason,
                    output_dir / f"attempt_{retry}",
                    self.tool_library.get_all_tools(),  # active tools for implementation
                    [t.name for t in self.tool_library.get_all_tools_including_deprecated()]  # all tool names for forbidden
                )
            if not merged_tool:
                logger.warning("Failed to create merged tool")
                all_attempts.append({
                    'retry': retry,
                    'tool': None,
                    'validation_result': None,
                    'failure_reason': 'Tool merge failed'
                })
                continue
            # 3. validate merged tool
            validation_result = self._validate_tool(
                merged_tool,
                examples_to_update,
                output_dir / f"validation_{retry}"
            )
            if merged_tool:
                attempt_info = {
                    'retry': retry,
                    'tool': {
                        'name': merged_tool.name,
                        'signature': merged_tool.signature,
                        'docstring': merged_tool.docstring,
                        'implementation': merged_tool.implementation
                    },
                    'validation_result': {
                        'execution_success_rate': validation_result.execution_success_rate,
                        'execution_successes': validation_result.execution_successes,
                        'execution_failures': validation_result.execution_failures,
                        'correctness_rate': validation_result.correctness_rate,
                        'correct_results': validation_result.correct_results,
                        'correctness_checks': validation_result.correctness_checks,
                        'total_examples': validation_result.total_examples,
                        'passed': validation_result.passed,
                        'execution_errors': [],
                        'correctness_failures': []
                    } if validation_result else None
                }
            else:   # error in tool parsing - store minimal info
                attempt_info = {
                    'retry': retry,
                    'tool': None,
                    'validation_result': None,
                    'failure_reason': 'Tool merge failed'
                }
            all_attempts.append(attempt_info)
            if validation_result.passed:
                logger.info(f"🎉 Merged tool {merged_tool.name} passed validation on attempt {retry + 1}")
                with open(output_dir / "final_status.json", 'w') as f:
                    json.dump({
                        'success': True,
                        'merged_tool': merged_tool.name,
                        'original_tools': [t.name for t in tools_to_merge],
                        'retry_count': retry,
                        'validation_rates': {
                            'execution': validation_result.execution_success_rate,
                            'correctness': validation_result.correctness_rate
                        },
                        'all_attempts': all_attempts
                    }, f, indent=2, ensure_ascii=False)
                return merged_tool, validation_result
            logger.warning(f"Validation failed - Execution: {validation_result.execution_success_rate:.2%}, "
                         f"Correctness: {validation_result.correctness_rate:.2%}")
        # all attempts failed
        with open(output_dir / "final_status.json", 'w') as f:  
            json.dump({
                'success': False,
                'original_tools': [t.name for t in tools_to_merge],
                'retry_count': self.config.max_merge_retries,
                'reason': 'Validation failed after all retries',
                'all_attempts': all_attempts
            }, f, indent=2, ensure_ascii=False)
        return None, None
    
    def _update_examples_after_merge(self, merged_tool, validation_result, output_dir):
        if not validation_result.successful_rewrites:
            return 0
        rewrites_to_apply = []
        for rewrite in validation_result.successful_rewrites:
            if rewrite['validation_passed'].get('execution', False):
                if self.config.require_correctness_for_update:  # only update examples that passed validation if divergent results
                    if rewrite['validation_passed'].get('correctness', False):
                        rewrites_to_apply.append(rewrite)
                else:
                    rewrites_to_apply.append(rewrite)
        if not rewrites_to_apply:
            logger.warning(f"No successful rewrites for merged tool {merged_tool.name}")
            return 0
        updated_count = self.example_library.update_examples_for_new_tool(
            tool_name=merged_tool.name,
            successful_rewrites=rewrites_to_apply,
            update_type='deduplication'
        )
        updates_dir = output_dir / "example_updates"
        self.example_library.save_update_history(updates_dir)
        self.state.examples = self.example_library.examples.copy()
        return updated_count