"""
program execution and validation components.
"""
import json
import signal
import runpy
import traceback
import sys
import textwrap
from pathlib import Path

from agents import CorrectnessJudge
from prompts import program_rewrite_prompt
from models import ValidationResult, ProgramCandidate
from logging_utils import get_logger, log_execution, log_tool_validation
from predefined_apis import PredefinedModules
from utils import LLMClient, extract_code, normalize_answer

class TimeoutError(Exception):
    pass

class ExecutionEngine:
    """executes programs sequentially"""
    def __init__(self, config):
        self.config = config
        self.timeout = config.execution_timeout
        self.save_exe_trace = config.save_exe_trace
        self.modules = PredefinedModules(config)
        self.logger = get_logger(__name__)
    
    def execute(self, candidate, tools, output_dir, image_path=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.modules.clear_trace()
        exec_file = output_dir / "program.py"
        result_file = output_dir / "result.json"
        trace_file = output_dir / "trace.html"
        error_file = output_dir / "error.txt"
        if image_path is None:
            image_path = Path(self.config.image_path) / candidate.question.image_filename
        if not image_path.exists():
            error_msg = f"Image file not found: {image_path}"
            self.logger.error(error_msg)
            with open(error_file, 'w') as f:
                f.write(f"Error: {error_msg}\n")
                f.write(f"Attempted path: {image_path}\n")
                f.write(f"Image filename from question: {candidate.question.image_filename}\n")
                f.write(f"Base image path: {self.config.image_path}\n")
            log_execution(
                candidate.question.question_index,
                candidate.candidate_index,
                success=False,
                error=error_msg
            )
            return {
                'namespace': {},
                'answer': None,
                'error': error_msg,
                'error_details': f"Attempted path: {image_path}"
            }
        # 1. write executable
        self._write_executable(exec_file, result_file, candidate.code, tools, str(image_path))  
        namespace = {
            "Path": Path,
            **self.modules.get_all()    # add predefined functions
        }
        # 2. execute program (will save result file)
        error_info = self._run_program(exec_file, namespace)
        # 3. save execution trace (optional --> saving these html files will take up quite some disk space | but good for examining the programs & debugging)
        if self.save_exe_trace:
            self._save_trace(trace_file, candidate, error_info.get('error') if error_info else None)
        # 4. write errors if any
        if error_info:
            with open(error_file, 'w') as f:
                f.write(f"Error: {error_info['error']}\n")
                f.write(f"Traceback:\n{error_info['traceback']}\n")
            log_execution(
                candidate.question.question_index,
                candidate.candidate_index,
                success=False,
                error=error_info['error']
            )
            return {
                'namespace': {},
                'answer': None,
                'error': error_info['error'],
                'error_details': error_info['traceback']
            }
        # 5. load results & log
        if result_file.exists():
            with open(result_file, 'r') as f:
                result_namespace = json.load(f)
            answer = result_namespace.get('final_result')    # extract answer formatted as per program generation prompt
            if answer is None:  # check if answer is none --> treat as execution error
                error_msg = "Program execution produced None result (final_result not set or is None)"
                with open(error_file, 'w') as f:
                    f.write(f"Error: {error_msg}\n")
                    f.write(f"Result namespace: {result_namespace}\n")
                log_execution(
                    candidate.question.question_index,
                    candidate.candidate_index,
                    success=False,
                    error=error_msg
                )
                return {
                    'namespace': result_namespace,
                    'answer': None,
                    'error': error_msg,
                    'error_details': 'The program did not set final_result or set it to None'
                }
            answer = normalize_answer(answer)
            log_execution(
                candidate.question.question_index,
                candidate.candidate_index,
                success=True,
                answer=answer
            )
            return {
                'namespace': result_namespace,
                'answer': answer,
                'error': None
            }
        else:
            error_msg = "No result file created"
            log_execution(
                candidate.question.question_index,
                candidate.candidate_index,
                success=False,
                error=error_msg
            )
            return {
                'namespace': {},
                'answer': None,
                'error': error_msg
            }
    
    def _write_executable(self, exec_file, result_file, code, tools, image_path_str):
        """write executable program file"""
        with open(exec_file, 'w') as f:
            # basic imports
            f.write("import json\n")
            f.write("import math\n")
            f.write("import numpy as np\n")
            f.write("from PIL import Image\n")
            f.write("from pathlib import Path\n\n")
            # set image path
            f.write(f"# Set image path\n")
            f.write(f"image_path = r'{image_path_str}'\n\n")
            # load image
            f.write("# Load image with error handling\n")
            f.write("try:\n")
            f.write("    image = Image.open(image_path).convert('RGB')\n")
            f.write("except Exception as e:\n")
            f.write("    print(f'Failed to load image {image_path}: {e}')\n")
            f.write("    raise\n\n")
            # write tool implementations (only learned tools, predefined functions already loaded into namespace)
            for tool in tools:
                if tool.level > 0:
                    f.write(f"# Tool: {tool.name} (Level {tool.level})\n")
                    f.write(f'{tool.docstring}\n')
                    implementation = tool.implementation.strip()
                    if implementation.startswith(f'def {tool.name}'):   # this is a complete function definition, skip the signature writing
                        self.logger.warning(f"Tool {tool.name} has duplicate function definition in implementation, using implementation only")
                        f.write(f"{implementation}\n\n")
                    else:
                        f.write(f"{tool.signature}\n")
                        # handle misaligned first line and second line
                        impl_lines = implementation.split("\n")
                        if not impl_lines[0].startswith('    '):
                            if impl_lines[1].startswith('    '):
                                implementation = '    ' + implementation
                                implementation = textwrap.dedent(implementation)
                        # indent implementation
                        indented = textwrap.indent(implementation, '    ')  
                        f.write(f"{indented}\n\n")
            # user program (generated by program generator)
            f.write("# USER PROGRAM\n")
            code = code.replace("exit()", "").replace("quit()", "") # remove accidental `exit()` `quit()`
            f.write(code)
            f.write("\n\n")
            f.write("# Ensure final_result is defined\n")   # ensure final_result exists
            f.write("if 'final_result' not in locals():\n")
            f.write("    final_result = None\n\n")
            f.write("# Save results\n") # save results & execution trace (namespaces)
            f.write("def is_json_serializable(obj):\n")
            f.write("    try:\n")
            f.write("        json.dumps(obj)\n")
            f.write("        return True\n")
            f.write("    except:\n")
            f.write("        return False\n\n")
            f.write("namespace = {\n")
            f.write("    k: v for k, v in globals().items()\n")
            f.write("    if is_json_serializable(v) and not k.startswith('_')\n")
            f.write("    and k not in ['image', 'Image', 'json', 'math', 'np', 'Path', 'image_path']\n")
            f.write("}\n\n")
            f.write(f"with open('{result_file}', 'w') as f:\n")
            f.write("    json.dump(namespace, f, ensure_ascii=False)\n")
    
    def _run_program(self, exec_file, namespace):
        """execute a program file with timeout"""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {self.timeout} seconds")
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
            runpy.run_path(str(exec_file), init_globals=namespace)
            signal.alarm(0)
            return None
        except Exception as e:
            signal.alarm(0)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            full_traceback = ''.join(tb_lines)
            return {
                'error': f"{exc_type.__name__}: {str(e)}",
                'traceback': full_traceback
            }
    
    def _save_trace(self, trace_file, candidate, error):
        """save full execution trace as html"""
        trace_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Execution Trace</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
        .error {{ color: red; font-weight: bold; }}
        .section {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Execution Trace</h1>
    
    <div class="section">
        <h2>Question</h2>
        <p>{candidate.question.question}</p>
        <p><strong>Image:</strong> {candidate.question.image_filename}</p>
    </div>
    
    <div class="section">
        <h2>Program</h2>
        <pre>{candidate.code}</pre>
    </div>
    
    <div class="section">
        <h2>Execution Trace</h2>
        {self.modules.get_trace() or '<p>No trace available</p>'}
    </div>
"""
        if error:
            trace_html += f"""
    <div class="section">
        <h2>Error</h2>
        <p class="error">{error}</p>
    </div>
"""
        trace_html += """
</body>
</html>
"""
        with open(trace_file, 'w') as f:
            f.write(trace_html)

class ProgramRewriter:
    """rewrites programs to use newly created tools"""    
    def __init__(self, config):
        self.config = config
        self.llm_client = LLMClient(config)
        self.logger = get_logger(__name__)
    
    def rewrite_with_tool(self, example, tool, output_dir, retry=0):
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            prompt = program_rewrite_prompt(
                example.program,
                tool.to_dict(),
                example.question.question
            )
            with open(output_dir / f"rewrite_prompt_retry{retry}.txt", 'w') as f:
                f.write(prompt)
            response = self.llm_client.generate(
                prompt,
                model=self.config.rewrite_model,
                temperature=self.config.temperature_rewrite
            )
            with open(output_dir / f"rewrite_response_retry{retry}.txt", 'w') as f:
                f.write(response)
            rewritten_code = extract_code(response, return_nan=False)
            if tool.name not in rewritten_code:
                self.logger.warning(f"Rewritten code does not use tool {tool.name}")
            if self._validate_python(rewritten_code):
                with open(output_dir / f"rewritten_program_retry{retry}.py", 'w') as f:
                    f.write(rewritten_code)
                return rewritten_code
            else:
                self.logger.warning(f"Invalid Python syntax in rewrite attempt {retry}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to rewrite program: {e}")
            return None

    def _validate_python(self, code):
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

class ExecutionValidator:
    """validates that rewritten programs execute successfully"""
    def __init__(self, config):
        self.config = config
        self.execution_engine = ExecutionEngine(config)
        self.rewriter = ProgramRewriter(config)
        self.logger = get_logger(__name__)
    
    def validate(self, tool, examples, all_tools, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        result = ValidationResult(
            tool_name=tool.name,
            cluster_id=tool.source_cluster_id or "unknown",
            total_examples=len(examples)
        )
        # calculate maximum allowed failures for early exit
        max_failures = int(len(examples) * (1 - self.config.min_execution_success_rate))
        self.logger.info(f"Maximum allowed execution failures: {max_failures} (for {self.config.min_execution_success_rate:.0%} success rate)")
        for i, example in enumerate(examples):
            example_dir = output_dir / f"example_{i}"
            example_dir.mkdir(exist_ok=True)
            self.logger.info(f"Validating example {i+1}/{len(examples)}")
            if result.execution_failures > max_failures:    # check if we should early exit
                self.logger.warning(f"Early exit: Already exceeded maximum allowed failures ({result.execution_failures} > {max_failures})")
                for j in range(i, len(examples)):   # mark remaining examples as not processed
                    result.execution_failures += 1
                    result.execution_errors.append({
                        'example_id': examples[j].id,
                        'error': 'Not executed due to early exit: Already exceeded maximum allowed execution failures',
                        'original_program': examples[j].program,
                        'rewritten_program': None,
                        'all_retry_errors': None
                    })
                break
            rewritten = None
            for retry in range(self.config.max_rewrite_retries):
                rewritten = self.rewriter.rewrite_with_tool(
                    example, 
                    tool, 
                    example_dir, 
                    retry
                )
                if not rewritten:
                    self.logger.warning(f"Rewrite attempt {retry+1} failed to generate code")
                    continue
                candidate = ProgramCandidate(
                    question=example.question,
                    code=rewritten,
                    candidate_index=0
                )
                all_tools_including_new = all_tools + [tool]
                exec_result = self.execution_engine.execute(
                    candidate,
                    all_tools_including_new,
                    example_dir / f"execution_retry_{retry}"
                )
                if not exec_result['error'] and exec_result['answer'] is not None:  # check for both error and none answer (treat as error)
                    result.execution_successes += 1
                    self.logger.info(f"  Example {i+1} execution successful")
                    original_namespace = example.execution_result if isinstance(example.execution_result, dict) else {'final_result': example.execution_result}
                    if isinstance(example.execution_result, dict):
                        original_answer = example.execution_result.get('final_result')
                    else:
                        original_answer = example.execution_result
                    self.logger.info(f"    Original answer: {original_answer}")
                    self.logger.info(f"    New answer: {exec_result['answer']}")
                    new_answer = normalize_answer(exec_result['answer'])
                    rewrite_info = {
                        'example_id': example.id,
                        'new_program': rewritten,
                        'original_answer': original_answer,
                        'new_answer': exec_result['answer'],
                        'validation_passed': {'execution': True},
                        'original_namespace': original_namespace,
                        'new_namespace': exec_result['namespace']   # store full namespaces for correctness validation
                    }
                    if isinstance(original_namespace, dict) and 'final_result' in original_namespace:
                        original_answer_normalized = normalize_answer(original_namespace['final_result'])
                    else:
                        original_answer_normalized = normalize_answer(original_answer)
                    if original_answer_normalized != new_answer:    # need correctness validation
                        result.correctness_checks += 1
                        result.correctness_details.append({
                            'example_id': example.id,
                            'original_answer': original_answer, 
                            'new_answer': exec_result['answer'],
                            'question': example.question.question,
                            'question_obj': example.question,
                            'original_program': example.program,
                            'rewritten_program': rewritten,
                            'original_namespace': original_namespace,
                            'new_namespace': exec_result['namespace']
                        })
                    else:
                        rewrite_info['validation_passed']['correctness'] = True
                    result.successful_rewrites.append(rewrite_info)
                    break
                else:
                    if exec_result['answer'] is None and not exec_result['error']:  # treat none answer as execution error
                        error_msg = "Program execution produced None result"
                    else:
                        error_msg = exec_result['error']
                    self.logger.warning(f"Execution attempt {retry+1} failed: {error_msg}")
            else:
                result.execution_failures += 1
                error_msg = "All rewritten program execution retries failed"
                original_namespace = example.execution_result if isinstance(example.execution_result, dict) else {'final_result': example.execution_result}
                result.execution_errors.append({
                    'example_id': example.id,
                    'error': error_msg,
                    'original_program': example.program,
                    'rewritten_program': rewritten if rewritten else None,
                    'all_retry_errors': error_msg,
                    'original_namespace': original_namespace, 
                    'attempted_namespace': exec_result.get('namespace', {}) if 'exec_result' in locals() else {}
                })
        log_tool_validation(
            stage="execution",
            success_rate=result.execution_success_rate,
            total_examples=result.total_examples,
            details={'successes': result.execution_successes,
                     'failures': result.execution_failures},
            config=self.config
        )
        return result

class CorrectnessValidator:
    """validates correctness when rewritten programs produce different answers"""
    def __init__(self, config):
        self.config = config
        self.correctness_judge = None
        self.tool_library = None
        self.current_tool_being_validated = None
        self.logger = get_logger(__name__)
    
    def set_tool_library(self, tool_library):
        """connect to tool library to access tool implementations"""
        self.tool_library = tool_library
        if self.correctness_judge is None:
            self.correctness_judge = CorrectnessJudge(self.config)
        self.correctness_judge.tool_library = tool_library
    
    def set_current_tool(self, tool):
        """set the tool currently being validated"""
        self.current_tool_being_validated = tool
    
    def validate(self, exec_result, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        if not exec_result.correctness_details:
            self.logger.info("No divergent results to validate")
            for rewrite in exec_result.successful_rewrites:
                if 'correctness' not in rewrite['validation_passed']:   # mark all non-divergent successful executions as passing correctness
                    rewrite['validation_passed']['correctness'] = True
            return exec_result
        valid_correctness_details = []
        for detail in exec_result.correctness_details:
            if detail.get('new_answer') is None:    # filter out cases where new answer is none
                self.logger.info(f"Skipping correctness validation for example {detail['example_id']} - execution failed (answer is None)")
                for rewrite in exec_result.successful_rewrites:
                    if rewrite['example_id'] == detail['example_id']:
                        rewrite['validation_passed']['correctness'] = False
                        break
            else:
                valid_correctness_details.append(detail)
        if not valid_correctness_details:
            self.logger.info("No valid divergent results to validate (all had None answers)")
            return exec_result
        self.logger.info(f"Validating {len(valid_correctness_details)} divergent results")
        for i, detail in enumerate(valid_correctness_details):
            judge_dir = output_dir / f"divergent_{i}"
            self.logger.info(f"  Checking divergent result {i+1}:")
            self.logger.info(f"    Question: {detail['question']}")
            self.logger.info(f"    Original answer: {detail['original_answer']}")
            self.logger.info(f"    New answer: {detail['new_answer']}")
            image_path = None
            if 'question_obj' in detail and hasattr(detail['question_obj'], 'image_filename'):
                question_obj = detail['question_obj']
                image_path = Path(self.config.image_path) / question_obj.image_filename
                if not image_path.exists():
                    self.logger.warning(f"Image file not found for correctness validation: {image_path}")
                    alt_paths = [
                        Path(self.config.image_path) / question_obj.image_filename.replace('\\', '/'),
                        Path(self.config.image_path) / Path(question_obj.image_filename).name
                    ]
                    for alt_path in alt_paths:
                        if alt_path.exists():
                            image_path = alt_path
                            break
                    else:
                        self.logger.error(f"Could not find image file in any expected location")
                        continue
            judgment = self.correctness_judge.judge_correctness(
                detail,
                image_path,
                judge_dir
            )
            self.logger.info(f"    Verdict: {judgment['verdict']}")
            if judgment['verdict'] == 'INCORRECT':
                self.logger.info(f"    Reasoning: {judgment['reasoning']}")
            elif judgment['verdict'] == 'CORRECT':
                exec_result.correct_results += 1
                for rewrite in exec_result.successful_rewrites:
                    if rewrite['example_id'] == detail['example_id']:
                        rewrite['validation_passed']['correctness'] = True
                        break
            else:   # mark as failed
                for rewrite in exec_result.successful_rewrites:
                    if rewrite['example_id'] == detail['example_id']:
                        rewrite['validation_passed']['correctness'] = False
                        break
            for item in exec_result.correctness_details:
                if item['example_id'] == detail['example_id']:
                    item['verdict'] = judgment['verdict']
                    item['reasoning'] = judgment['reasoning']
                    break
        divergent_rate = exec_result.correctness_rate
        overall_rate = exec_result.overall_correctness_rate
        self.logger.info(f"Correctness validation complete:")
        self.logger.info(f"  - Among divergent results -- correctness rate: {divergent_rate:.2%} ({exec_result.correct_results}/{exec_result.correctness_checks})")
        self.logger.info(f"  - Among all examples -- overall correctness rate: {overall_rate:.2%} (considering all {exec_result.total_examples} examples)")
        log_tool_validation(
            "correctness",
            overall_rate, 
            exec_result.total_examples,
            details={
                'correct': exec_result.correct_results,
                'incorrect': exec_result.correctness_checks - exec_result.correct_results,
                'divergent_count': exec_result.correctness_checks,
                'non_divergent_count': exec_result.execution_successes - exec_result.correctness_checks,
                'overall_correct': int(overall_rate * exec_result.total_examples)
            },
            config=self.config
        )
        return exec_result