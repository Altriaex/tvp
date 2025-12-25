"""
logging utilities for the pipeline.
"""
import logging as log_module
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
_stats = defaultdict(lambda: defaultdict(int))

class StructuredFormatter(log_module.Formatter):
    def __init__(self):
        super().__init__(LOG_FORMAT, DATE_FORMAT)
        self.last_section = None
    
    def format(self, record):
        if record.name in ['httpx', 'urllib3', 'requests']:
            return ""
        section = getattr(record, 'section', None)
        output = ""
        if section and section != self.last_section:
            self.last_section = section
            output += f"\n{'='*80}\n"
            output += f"=== {section.upper()} ===\n"
            output += f"{'='*80}\n"
        formatted = super().format(record)
        if record.levelname == 'ERROR':
            formatted = f"❌ {formatted}"
        elif record.levelname == 'WARNING':
            formatted = f"⚠️ {formatted}"
        elif hasattr(record, 'success'):
            if record.success:
                formatted = f"✅ {formatted}"
            else:
                formatted = f"❌ {formatted}"
        return output + formatted

class JsonFileHandler(log_module.Handler):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
    
    def emit(self, record):
        try:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            if hasattr(record, 'event_data'):
                log_data['event_data'] = record.event_data
            with open(self.filename, 'a') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        except Exception:
            self.handleError(record)

def setup_logging(config):
    """setup comprehensive logging for TVP"""
    log_dir = config.results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    root_logger = log_module.getLogger()
    root_logger.setLevel(log_module.DEBUG if config.debug else log_module.INFO)
    root_logger.handlers.clear()
    console_handler = log_module.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredFormatter())
    console_handler.setLevel(log_module.INFO)
    root_logger.addHandler(console_handler)
    file_handler = log_module.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(log_module.Formatter(LOG_FORMAT, DATE_FORMAT))
    file_handler.setLevel(log_module.DEBUG)
    root_logger.addHandler(file_handler)
    error_handler = log_module.FileHandler(log_dir / "errors.log", mode='a')
    error_handler.setFormatter(log_module.Formatter(LOG_FORMAT, DATE_FORMAT))
    error_handler.setLevel(log_module.ERROR)
    root_logger.addHandler(error_handler)
    json_handler = JsonFileHandler(log_dir / "events.jsonl") # json logger for structured events
    json_handler.setLevel(log_module.INFO)
    pipeline_logger = log_module.getLogger('pipeline')
    pipeline_logger.addHandler(json_handler)
    log_module.getLogger('httpx').setLevel(log_module.WARNING)
    log_module.getLogger('urllib3').setLevel(log_module.WARNING)
    log_module.getLogger('requests').setLevel(log_module.WARNING)
    log_module.getLogger('openai').setLevel(log_module.WARNING)
    log_module.getLogger('utils').setLevel(log_module.WARNING)
    if not sys.stdout.isatty():
        os.environ['TQDM_DISABLE'] = '1'
    logger = log_module.getLogger(__name__)
    logger.info(f"🚀 Pipeline initialized - Run ID: {config.run_id}", extra={'section': 'initialization'}) # log startup

def log_config(config):
    """log pipeline config"""
    logger = log_module.getLogger('pipeline')
    config_dict = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in config.__dict__.items()
        if not k.startswith('_')
    }
    logger.info(f"Configuration loaded:", extra={'section': 'configuration'})
    for k,v in config_dict.items():
        logger.info(f"  {k}: {v}")
    logger.info("Pipeline configuration", extra={'event_data': {
        'event': 'config_loaded',
        'config': config_dict
    }})
    config_path = config.results_dir / "metadata.json" # save to file for resumability
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def log_iteration_start(iteration, total_iterations):
    logger = log_module.getLogger('pipeline')
    logger.info(f"🔄 Starting iteration {iteration} of {total_iterations}", 
                extra={'section': f'iteration {iteration}', 'event_data': {
                    'event': 'iteration_start',
                    'iteration': iteration,
                    'total_iterations': total_iterations
                }})

def log_question_processing(question_index, question_text):
    logger = log_module.getLogger('pipeline')
    logger.info(
        f"Processing question {question_index}: {question_text}",
        extra={'event_data': {
            'event': 'question_start',
            'question_index': question_index,
            'question_text': question_text
        }}
    )

def log_question_complete(question_index, example_added, quality_rating=None):
    logger = log_module.getLogger('pipeline')
    status = "✅ Added to library" if example_added else "⏭️ Skipped"
    quality_text = f" (quality: {quality_rating:.1f})" if quality_rating else ""
    logger.info(
        f"Question {question_index} completed - {status}{quality_text}",
        extra={'success': example_added}
    )

def log_abstraction_interval(questions_processed, examples_count, tools_count):
    logger = log_module.getLogger('pipeline')
    logger.info(
        f"🔮 Abstraction interval at {questions_processed} questions - {examples_count} examples, {tools_count} tools",
        extra={'section': 'abstraction', 'event_data': {
            'event': 'abstraction_interval_start',
            'questions_processed': questions_processed,
            'examples_count': examples_count,
            'tools_count': tools_count
        }}
    )

def log_retrieval(question_id, retrieved_count, similarities):
    logger = log_module.getLogger('library')
    if retrieved_count > 0:
        logger.info(
            f"Retrieved {retrieved_count} examples (similarity: {max(similarities):.3f} - {min(similarities):.3f})",
            extra={'event_data': {
                'event': 'examples_retrieved',
                'question_id': question_id,
                'retrieved_count': retrieved_count,
                'similarities': similarities,
                'max_similarity': max(similarities),
                'min_similarity': min(similarities)
            }}
        )

def log_program_generation(question_id, candidate_index, success, error=None):
    logger = log_module.getLogger('agents')
    _stats['program_generation']['total'] += 1
    if success:
        _stats['program_generation']['success'] += 1
    else:
        _stats['program_generation']['failed'] += 1
    logger.info("Program generation")
    logger.debug(
        f"Generated program {candidate_index} for question {question_id}",
        extra={'event_data': {
            'event': 'program_generated',
            'question_id': question_id,
            'candidate_index': candidate_index,
            'success': success,
            'error': error
        }}
    )

def log_execution(question_id, candidate_index, success, answer=None, error=None):
    logger = log_module.getLogger('execution')
    _stats['execution']['total'] += 1
    if success:
        _stats['execution']['success'] += 1
    else:
        _stats['execution']['failed'] += 1
        if error:
            error_type = 'timeout' if 'timeout' in error.lower() else 'runtime_error'
            _stats['execution_errors'][error_type] += 1
    event_data = {
        'event': 'program_executed',
        'question_id': question_id,
        'candidate_index': candidate_index,
        'success': success,
        'answer': str(answer) if answer is not None else None,
        'error': error
    }
    if success:
        logger.debug(f"Program {candidate_index} executed successfully - Answer: {answer}",
                    extra={'event_data': event_data})
    else:
        logger.debug(f"Execution failed for program {candidate_index}, question {question_id}: {error}",
                    extra={'event_data': event_data})

def log_execution_summary(total_candidates, successful):
    logger = log_module.getLogger('pipeline')
    if successful == 0:
        logger.warning(f"⚠️ All {total_candidates} program executions failed")
    else:
        logger.info(f"Execution summary: {successful}/{total_candidates} generated programs succeeded")

def log_quality_judgment(question_id, candidate_index, quality_rating, reasoning):
    logger = log_module.getLogger('agents')
    logger.debug(
        f"Judged program {candidate_index}: {quality_rating:.1f}/10",
        extra={'event_data': {
            'event': 'quality_judged',
            'question_id': question_id,
            'candidate_index': candidate_index,
            'quality_rating': quality_rating,
            'reasoning_preview': reasoning
        }}
    )

def log_example_added(example_id, quality_rating, tools_used):
    logger = log_module.getLogger('library')
    _stats['examples']['added'] += 1
    _stats['examples']['total_quality'] += quality_rating
    logger.info(
        f"📚 Added example {example_id} to library (quality: {quality_rating:.1f}, tools: {len(tools_used)})",
        extra={'success': True, 'event_data': {
            'event': 'example_added',
            'example_id': example_id,
            'quality_rating': quality_rating,
            'tools_used': tools_used,
            'tool_count': len(tools_used)
        }}
    )

def log_example_updated(example_id, update_type, reason, old_tools, new_tools):
    logger = log_module.getLogger('library')
    _stats['example_updates']['total'] += 1
    _stats['example_updates'][update_type] += 1
    if update_type == 'abstraction':
        icon = "🔄"
    elif update_type == 'deduplication':
        icon = "🔀"
    elif update_type == 'iteration_update':
        icon = "📝"
    else:
        icon = "📋"
    logger.info(
        f"{icon} Updated example {example_id}: {reason} (tools: {len(old_tools)} → {len(new_tools)})",
        extra={'success': True, 'event_data': {
            'event': 'example_updated',
            'example_id': example_id,
            'update_type': update_type,
            'reason': reason,
            'old_tools': old_tools,
            'new_tools': new_tools
        }}
    )

def log_clustering_result(interval_num, groups_found, clusters_created):
    logger = log_module.getLogger('library')
    logger.info(
        f"Clustering for interval #{interval_num}: {groups_found} initial groups → {clusters_created} final clusters",
        extra={'event_data': {
            'event': 'clustering_complete',
            'interval_number': interval_num,
            'initial_groups': groups_found,
            'clusters_created': clusters_created
        }}
    )

def log_abstraction_start(clusters_to_abstract):
    logger = log_module.getLogger('pipeline')
    logger.info(
        f"🔧 Starting tool abstraction from {clusters_to_abstract} clusters",
        extra={'event_data': {
            'event': 'abstraction_start',
            'clusters_to_abstract': clusters_to_abstract
        }}
    )

def log_tool_abstracted(tool_name, tool_level, examples_count, success_rate):
    logger = log_module.getLogger('agents')
    _stats['abstraction']['successful'] += 1
    logger.info(
        f"💎 Created tool: {tool_name} (level {tool_level}, {examples_count} examples, {success_rate:.0%} success)",
        extra={'success': True, 'event_data': {
            'event': 'tool_abstracted',
            'tool_name': tool_name,
            'tool_level': tool_level,
            'examples_count': examples_count,
            'success_rate': success_rate
        }}
    )

def log_tool_abstraction(cluster_id, tool_name, success, retry_count, validation_scores=None):
    logger = log_module.getLogger('agents')
    _stats['abstraction']['attempts'] += 1
    if success:
        _stats['abstraction']['successful'] += 1
    else:
        _stats['abstraction']['failed'] += 1
    event_data = {
        'event': 'tool_abstraction',
        'cluster_id': cluster_id,
        'tool_name': tool_name if success else None,
        'success': success,
        'retry_count': retry_count
    }
    if validation_scores:
        event_data['validation_scores'] = validation_scores
    if success:
        logger.info(f"🔧 Created tool: {tool_name}", extra={'success': True})
    else:
        logger.warning(f"Failed to abstract tool from cluster {cluster_id} after {retry_count} retries")

def log_tool_validation(stage, success_rate, total_examples, details=None, config=None):
    logger = log_module.getLogger('validation')
    min_execution_rate = config.min_execution_success_rate
    min_correctness_rate = config.min_correctness_rate
    if stage == "execution":
        status = "✅" if success_rate >= min_execution_rate else "❌"
        logger.info(f"{status} Stage 1 validation: {success_rate:.0%} execution success ({details.get('successes', 0)}/{total_examples})")
    elif stage == "correctness":
        divergent_count = details.get('divergent_count', 0) if details else 0
        non_divergent_count = details.get('non_divergent_count', 0) if details else 0
        if divergent_count > 0:
            correct_divergent = details.get('correct', 0)
            overall_correct = correct_divergent + non_divergent_count
            correct_rate = overall_correct / total_examples
            status = "✅" if correct_rate >= min_correctness_rate else "❌"
            logger.info(f"{status} Stage 2 validation: {correct_divergent} among {divergent_count} divergent results -- overall: {correct_rate:.0%} among all {total_examples} results ({correct_divergent}+{non_divergent_count}/{total_examples})")
        else:
            logger.info("✅ Stage 2 validation: No divergent results")

def log_deduplication_start(tools_count):
    logger = log_module.getLogger('pipeline')
    logger.info(
        f"🔀 Starting tool deduplication with {tools_count} learned tools",
        extra={'event_data': {
            'event': 'deduplication_start',
            'tools_count': tools_count
        }}
    )

def log_duplicates_found(merge_groups_count):
    logger = log_module.getLogger('agents')
    logger.info(
        f"🔍 Found {merge_groups_count} groups of similar tools to merge",
        extra={'event_data': {
            'event': 'duplicates_found',
            'merge_groups_count': merge_groups_count
        }}
    )

def log_tool_merged(original_tools, merged_tool_name, examples_updated):
    logger = log_module.getLogger('agents')
    logger.info(
        f"🔗 Successfully merged {len(original_tools)} tools: {', '.join(original_tools)} --into--> {merged_tool_name} & updated {examples_updated} examples",
        extra={'success': True, 'event_data': {
            'event': 'tool_merged',
            'original_tools': original_tools,
            'merged_tool_name': merged_tool_name,
            'examples_updated': examples_updated
        }}
    )

def log_tool_deprecation(tool_name, reason, metrics):
    logger = log_module.getLogger('library')
    logger.warning(
        f"🗑️ Deprecated tool {tool_name}: {reason}",
        extra={'event_data': {
            'event': 'tool_deprecated',
            'tool_name': tool_name,
            'reason': reason,
            'metrics': metrics
        }}
    )

def log_example_updates_summary(updated_count, total_examples, update_type):
    logger = log_module.getLogger('library')
    if updated_count > 0:
        logger.info(
            f"📊 Updated {updated_count}/{total_examples} examples for {update_type}",
            extra={'success': True, 'event_data': {
                'event': 'example_updates_summary',
                'updated_count': updated_count,
                'total_examples': total_examples,
                'update_type': update_type
            }}
        )

def log_iteration_summary(iteration):
    logger = log_module.getLogger('pipeline')
    logger.info(
        f"Iteration {iteration} complete",
        extra={'section': f'iteration {iteration} summary', 'event_data': {
            'event': 'iteration_complete',
            'iteration': iteration,
            'stats': dict(_stats)
        }}
    )

def log_resume_info(current_iteration, questions_processed):
    logger = log_module.getLogger('pipeline')
    if questions_processed > 0:
        logger.info(
            f"📂 Resuming from iteration {current_iteration}, {questions_processed} total questions processed",
            extra={'event_data': {
                'event': 'resume',
                'current_iteration': current_iteration,
                'questions_processed': questions_processed
            }}
        )
    else:
        logger.info("🆕 Starting fresh run")

def log_final_summary(questions_processed, examples_collected, tools_total, execution_time):
    logger = log_module.getLogger('pipeline')
    logger.info(
        f"\n{'='*80}\n=== PIPELINE COMPLETE ===\n{'='*80}",
        extra={'section': 'final summary'}
    )
    logger.info(f"📊 Final Statistics:")
    logger.info(f"  - Questions processed: {questions_processed}")
    logger.info(f"  - Examples collected: {examples_collected}")
    logger.info(f"  - Tools created: {tools_total}")
    logger.info(f"  - Execution time: {execution_time}")
    if _stats:
        logger.info("\n📈 Detailed Metrics:")
        for category, metrics in _stats.items():
            logger.info(f"\n  {category.upper()}:")
            for metric, value in metrics.items():
                logger.info(f"    - {metric}: {value}")

def get_logger(name):
    return log_module.getLogger(name)
