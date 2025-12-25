"""
llm agentic components in TVP
"""
import re
import json
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO
import traceback

from models import Tool, Cluster, ProgramCandidate
from prompts import *
from utils import LLMClient, extract_code, extract_tools_from_program
from logging_utils import get_logger, log_quality_judgment

class BaseAgent:
    """base class for all llm agents"""
    def __init__(self, config):
        self.config = config
        self.llm_client = LLMClient(config)
        self.logger = get_logger(self.__class__.__name__)
    
    def _save_prompt(self, prompt, output_dir, filename="prompt.txt"):
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / filename, 'w') as f:
            f.write(prompt)
    
    def _save_response(self, response, output_dir, filename="response.txt"):
        with open(output_dir / filename, 'w') as f:
            f.write(response)
    
    def _save_json(self, data, output_dir, filename="data.json"):
        with open(output_dir / filename, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

class ProgramGenerator(BaseAgent):
    """generates program candidates for questions"""
    def __init__(self, config):
        super().__init__(config)
        self.model = config.program_model
    
    def generate_single(self, task):
        question = task['question']
        examples = task['examples']
        tools = task['tools']
        index = task['index']
        output_dir = task['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            tools_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in tools]
            examples_dicts = [ex.to_dict() if hasattr(ex, 'to_dict') else ex for ex in examples]
            prompt = program_generation_prompt(
                question.question,
                examples_dicts,
                tools_dicts
            )
            self._save_prompt(prompt, output_dir, f"prompt_{index}.txt")
            response = self.llm_client.generate(
                prompt,
                model=self.model,
                temperature=self.config.temperature_program
            )
            self._save_response(response, output_dir, f"response_{index}.txt")
            code = extract_code(response, return_nan=True)
            if code:
                candidate = ProgramCandidate(
                    question=question,
                    code=code,
                    candidate_index=index
                )
                self._save_json({
                    'question_index': question.question_index,
                    'candidate_index': index,
                    'code': code,
                    'model': self.model
                }, output_dir, f"candidate_{index}.json")
                return candidate
            else:
                self.logger.warning(f"Failed to extract code for candidate {index}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to generate program: {e}")
            self.logger.debug(traceback.format_exc())
            return None

class QualityJudge(BaseAgent):
    """judges program quality with visual evidence"""
    def __init__(self, config):
        super().__init__(config)
        self.model = config.judge_model
    
    def judge_single(self, task):
        candidate = task['candidate']
        output_dir = task['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            tools = task.get('tools', [])
            tools_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in tools]
            execution_result = candidate.execution_result
            execution_error = candidate.execution_error
            prompt = quality_judge_prompt(
                candidate.question.question,
                candidate.code,
                execution_result,
                execution_error,
                tools_dicts
            )
            self._save_prompt(prompt, output_dir)
            image_path = Path(self.config.image_path) / candidate.question.image_filename
            with Image.open(image_path) as img:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
            response = self.llm_client.generate_with_image(
                prompt,
                img_base64,
                model=self.model,
                temperature=self.config.temperature_judge
            )
            self._save_response(response, output_dir)
            rating, reasoning = self._parse_judgment(response)
            candidate.quality_rating = rating
            candidate.quality_reasoning = reasoning
            self._save_json({
                'question_index': candidate.question.question_index,
                'candidate_index': candidate.candidate_index,
                'quality_rating': rating,
                'quality_reasoning': reasoning,
                'answer': candidate.answer,
                'model': self.model
            }, output_dir, "judgment.json")
            log_quality_judgment(
                candidate.question.question_index,
                candidate.candidate_index,
                rating,
                reasoning
            )
            return candidate
        except Exception as e:
            self.logger.error(f"Failed to judge program: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def _parse_judgment(self, response):
        """parse quality rating and reasoning from response formatted as per quality judge prompt"""
        rating_match = re.search(r'<rating>([\d.]+)</rating>', response)
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        if rating_match:
            rating = min(max(float(rating_match.group(1)), 1.0), 10.0)
        else:
            rating = 0.0
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            reasoning = "No reasoning provided"
        return rating, reasoning

class ClusterAnalyzer(BaseAgent):
    """analyzes example clusters for abstraction potential"""
    def __init__(self, config):
        super().__init__(config)
        self.model = config.cluster_model
    
    def analyze_group(self, examples, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            examples_dicts = [ex.to_dict() for ex in examples]
            prompt = cluster_analysis_prompt(examples_dicts)
            self._save_prompt(prompt, output_dir)
            response = self.llm_client.generate(
                prompt,
                model=self.model,
                temperature=self.config.temperature_cluster
            )
            self._save_response(response, output_dir)
            clusters = self._parse_clusters(response, self.config.min_cluster_size)
            self._save_json({
                'group_size': len(examples),
                'clusters_found': len(clusters),
                'clusters': [c.to_dict() for c in clusters]
            }, output_dir, "cluster_analysis.json")
            return clusters
        except Exception as e:
            self.logger.error(f"Failed to analyze cluster: {e}")
            return []
    
    def _parse_clusters(self, response, min_cluster_size):
        """parse clusters from response formatted as per cluster analysis prompt"""
        clusters = []
        cluster_matches = re.finditer(r'<cluster>(.*?)</cluster>', response, re.DOTALL)
        for i, match in enumerate(cluster_matches):
            cluster_text = match.group(1)
            example_ids_match = re.search(r'<example_ids>(.*?)</example_ids>', cluster_text, re.DOTALL)
            pattern_match = re.search(r'<pattern>(.*?)</pattern>', cluster_text, re.DOTALL)
            params_match = re.search(r'<parameters>(.*?)</parameters>', cluster_text, re.DOTALL)
            potential_match = re.search(r'<abstraction_potential>([\d.]+)</abstraction_potential>', cluster_text)
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', cluster_text, re.DOTALL)
            if example_ids_match and pattern_match and potential_match:
                example_ids_text = example_ids_match.group(1).strip()
                example_ids = [id.strip() for id in example_ids_text.split(',')]
                if min_cluster_size and len(example_ids) < min_cluster_size:    # check minimum cluster size
                    self.logger.info(f"Skipping LLM-analyzed cluster with only {len(example_ids)} examples (min: {min_cluster_size})")
                    continue
                cluster = Cluster(
                    id=f"cluster_{i}",
                    pattern=pattern_match.group(1).strip(),
                    parameters=params_match.group(1).strip() if params_match else "",
                    abstraction_potential=float(potential_match.group(1)),
                    reasoning=reasoning_match.group(1).strip() if reasoning_match else "",
                    example_ids=example_ids
                )
                clusters.append(cluster)
        return clusters

class ToolAbstractor(BaseAgent):
    """creates abstracted tools from example clusters"""
    def __init__(self, config):
        super().__init__(config)
        self.model = config.abstraction_model
    
    def abstract_from_cluster(self, cluster, examples, existing_tools, all_tools_including_deprecated, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            cluster_info = {
                'pattern': cluster.pattern,
                'parameters': cluster.parameters,
                'abstraction_potential': cluster.abstraction_potential,
                'reasoning': cluster.reasoning
            }
            example_dicts = []
            for ex in examples:
                example_dicts.append({
                    'id': ex.id,
                    'question': ex.question.question,
                    'answer': ex.question.answer,
                    'answer_type': ex.question.answer_type,
                    'program': ex.program,
                    'quality_rating': ex.quality_rating,
                    'execution_result': ex.execution_result  # include full execution trace
                })
            tool_dicts = [t.to_dict() for t in existing_tools]  # active tools for implementation
            used_tool_names = [t.name for t in all_tools_including_deprecated]  # get all used tool names (including deprecated) for forbidden names
            prompt = tool_abstraction_prompt(
                cluster_info,
                example_dicts,
                tool_dicts,
                used_tool_names
            )
            prompt_filename = "abstraction_prompt.txt"
            response_filename = "abstraction_response.txt"
            self._save_prompt(prompt, output_dir, prompt_filename)
            response = self.llm_client.generate(
                prompt,
                model=self.model,
                temperature=self.config.temperature_abstraction
            )
            self._save_response(response, output_dir, response_filename)
            tool = self._parse_tool(response, existing_tools)
            if tool:
                self._save_json({
                    'name': tool.name,
                    'level': tool.level,
                    'signature': tool.signature,
                    'docstring': tool.docstring,
                    'implementation': tool.implementation,
                    'source_cluster_id': cluster.id,
                    'refined': False
                }, output_dir, "tool_definition.json")
                self.logger.info(f"Abstracted tool {tool.name} at level {tool.level}")
            else:
                self.logger.warning("Failed to parse tool from response")
            return tool
        except Exception as e:
            self.logger.error(f"Failed to abstract tool: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def _parse_tool(self, response, existing_tools):
        """parse tool from llm response formatted as per tool abstraction prompt"""
        try:
            match = re.search(r'<function_name>(.*?)</function_name>', response, re.DOTALL)
            if not match:
                self.logger.warning("No function name found")
                return None
            name = match.group(1).strip()
            if not name.startswith('_'):
                name = f'_{name}'
            match = re.search(r'<signature>(.*?)</signature>', response, re.DOTALL)
            if not match:
                self.logger.warning("No signature found")
                return None
            signature = match.group(1).strip()
            if not signature.startswith(f'def {name}'):
                signature = re.sub(r'def\s+\w+', f'def {name}', signature)
            match = re.search(r'<docstring>(.*?)</docstring>', response, re.DOTALL)
            if not match:
                self.logger.warning("No docstring found")
                return None
            docstring = match.group(1).strip()
            match = re.search(r'<implementation>(.*?)</implementation>', response, re.DOTALL)
            if not match:
                self.logger.warning("No implementation found")
                return None
            implementation = match.group(1).strip()
            level = self._determine_tool_level(implementation, existing_tools)
            tool = Tool(
                name=name,
                level=level,
                signature=signature,
                docstring=docstring,
                implementation=implementation
            )
            return tool
        except Exception as e:
            self.logger.error(f"Failed to parse tool: {e}")
            return None
    
    def _determine_tool_level(self, implementation, existing_tools):
        max_level = 0
        function_calls = extract_tools_from_program(implementation) # extract all function calls from implementation (including predefined=level0)
        for call in function_calls:
            for tool in existing_tools:
                if tool.name == call:
                    max_level = max(max_level, tool.level)
        return max_level + 1

class ToolDeduplicator(BaseAgent):
    """identifies and merges duplicate tools"""
    def __init__(self, config):
        super().__init__(config)
        self.model = config.deduplication_model
    
    def find_duplicates(self, tools, output_dir):
        """find groups of duplicate/similar tools"""
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            tool_dicts = [t.to_dict() for t in tools]
            prompt = tool_deduplication_prompt(tool_dicts)
            self._save_prompt(prompt, output_dir)
            response = self.llm_client.generate(
                prompt,
                model=self.model,
                temperature=self.config.temperature_deduplication
            )
            self._save_response(response, output_dir)
            all_merge_groups = self._parse_merge_groups(response, tools, filter_by_threshold=False)
            with open(output_dir / "all_merge_groups.json", 'w') as f:
                json.dump({
                    'total_groups_found': len(all_merge_groups),
                    'threshold': self.config.deduplication_threshold,
                    'groups': all_merge_groups
                }, f, indent=2, ensure_ascii=False)
            merge_groups = [g for g in all_merge_groups if g['similarity_score'] >= self.config.deduplication_threshold]    # filter by threshold
            self._save_json({
                'total_tools': len(tools),
                'merge_groups_found': len(merge_groups),
                'merge_groups': merge_groups
            }, output_dir, "duplicate_analysis.json")
            return merge_groups
        except Exception as e:
            self.logger.error(f"Failed to find duplicates: {e}")
            return []
    
    def merge_tools(self, tools, merge_strategy, merge_strategy_reason, output_dir, all_existing_tools=None, used_tool_names=None):
        """merge similar tools into a single function"""
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            tool_dicts = [t.to_dict() for t in tools]
            if used_tool_names is None:
                if all_existing_tools:
                    used_tool_names = [t.name for t in all_existing_tools]
                else:
                    used_tool_names = []
            prompt = tool_merge_implementation_prompt(
                tool_dicts,
                merge_strategy,
                merge_strategy_reason,
                [t.to_dict() for t in all_existing_tools] if all_existing_tools else [],  # active tools
                used_tool_names  # all tool names (including deprecated) for forbidden
            )
            self._save_prompt(prompt, output_dir)
            response = self.llm_client.generate(
                prompt,
                model=self.config.merge_model,
                temperature=self.config.temperature_merge
            )
            self._save_response(response, output_dir)
            merged_tool = self._parse_merged_tool(response, tools)
            if merged_tool:
                self._save_json({
                    'original_tools': [t.name for t in tools],
                    'merged_tool': merged_tool.to_dict(),
                    'merge_strategy': merge_strategy
                }, output_dir, "merged_tool.json")
                self.logger.info(f"To merge {len(tools)} tools: {', '.join([t.name for t in tools])} --into--> {merged_tool.name}")
            return merged_tool
        except Exception as e:
            self.logger.error(f"Failed to merge tools: {e}")
            return None

    def _parse_merge_groups(self, response, tools, filter_by_threshold=True):
        """parse merge groups from response formatted as per tool deduplication prompt"""
        groups = []
        group_matches = re.finditer(r'<merge_group>(.*?)</merge_group>', response, re.DOTALL)
        for match in group_matches:
            group_text = match.group(1)
            indices_match = re.search(r'<tool_indices>(.*?)</tool_indices>', group_text)
            score_match = re.search(r'<similarity_score>([\d.]+)</similarity_score>', group_text)
            strategy_match = re.search(r'<merge_strategy>(.*?)</merge_strategy>', group_text, re.DOTALL)
            name_match = re.search(r'<recommended_name>(.*?)</recommended_name>', group_text)
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', group_text, re.DOTALL)
            if indices_match and score_match:
                similarity_score = float(score_match.group(1))
                if filter_by_threshold and similarity_score < self.config.deduplication_threshold:  # filter by deduplication threshold
                    self.logger.info(f"Skipping merge group with similarity {similarity_score:.2f} < threshold {self.config.deduplication_threshold}")
                    continue
                indices = [int(i.strip()) - 1 for i in indices_match.group(1).split(',')]
                tool_names = []
                for idx in indices:
                    if 0 <= idx < len(tools):
                        tool_names.append(tools[idx].name)
                if len(tool_names) >= 2:
                    groups.append({
                        'tool_names': tool_names,
                        'similarity_score': similarity_score,
                        'merge_strategy': strategy_match.group(1).strip() if strategy_match else "",
                        'recommended_name': name_match.group(1).strip() if name_match else "_merged_tool",
                        'reasoning': reasoning_match.group(1).strip() if reasoning_match else ""
                    })
        return groups
    
    def _parse_merged_tool(self, response, original_tools):
        """parse merged tool from response formatted as per tool merge implementation prompt"""
        try:
            name_match = re.search(r'<function_name>(.*?)</function_name>', response, re.DOTALL)
            sig_match = re.search(r'<signature>(.*?)</signature>', response, re.DOTALL)
            doc_match = re.search(r'<docstring>(.*?)</docstring>', response, re.DOTALL)
            impl_match = re.search(r'<implementation>(.*?)</implementation>', response, re.DOTALL)
            if not all([name_match, sig_match, doc_match, impl_match]):
                self.logger.warning("Missing required fields in merged tool response")
                return None
            name = name_match.group(1).strip()
            if not name.startswith('_'):
                name = f'_{name}'
            signature = sig_match.group(1).strip()
            if not signature.startswith(f'def {name}'):
                signature = re.sub(r'def\s+\w+', f'def {name}', signature)
            implementation = impl_match.group(1).strip()
            level = self._determine_merged_tool_level(original_tools)
            total_usage = sum(t.usage_count for t in original_tools)
            tool = Tool(
                name=name,
                level=level,
                signature=signature,
                docstring=doc_match.group(1).strip(),
                implementation=implementation,
                usage_count=total_usage,
                merged_from=[t.name for t in original_tools]
            )
            return tool
        except Exception as e:
            self.logger.error(f"Failed to parse merged tool: {e}")
            return None
    
    def _determine_merged_tool_level(self, original_tools):
        max_level = 0
        for tool in original_tools:
            max_level = max(max_level, tool.level)  # merged tool level = the same level as the highest original tool
        return max_level

class CorrectnessJudge(BaseAgent):
    """judges correctness when answers differ during validation (both abstraction & deduplication)"""
    def __init__(self, config):
        super().__init__(config)
        self.model = config.correctness_model
        self.tool_library = None
    
    def judge_correctness(self, detail, image_path, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            tool_implementations = self._extract_tool_implementations(
                detail.get('original_program', ''),
                detail.get('rewritten_program', '')
            )
            original_namespace = detail.get('original_namespace')
            new_namespace = detail.get('new_namespace')
            prompt = correctness_validation_prompt(
                detail['question'],
                detail['original_answer'],
                detail['new_answer'],
                detail['original_program'],
                detail['rewritten_program'],
                tool_implementations,
                original_namespace,
                new_namespace
            )
            self._save_prompt(prompt, output_dir)
            with Image.open(image_path) as img:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
            response = self.llm_client.generate_with_image(
                prompt,
                img_base64,
                model=self.model,
                temperature=self.config.temperature_correctness
            )
            self._save_response(response, output_dir)
            verdict, reasoning = self._parse_verdict(response)
            result = {
                'example_id': detail['example_id'],
                'verdict': verdict,
                'reasoning': reasoning,
                'original_answer': detail['original_answer'],
                'new_answer': detail['new_answer'],
                'question': detail['question']
            }
            self._save_json(result, output_dir, "correctness_judgment.json")
            return result
        except Exception as e:
            self.logger.error(f"Failed to judge correctness: {e}")
            return {
                'example_id': detail['example_id'],
                'verdict': 'ERROR',
                'reasoning': str(e)
            }
    
    def _extract_tool_implementations(self, original_program, rewritten_program):
        if not self.tool_library:
            return []
        implementations = []
        seen = set()
        for program_source, source_label in [(original_program, 'original'), (rewritten_program, 'rewritten')]:
            matches = extract_tools_from_program(program_source)   # including both learned tools & predefined
            for tool_name in matches:
                if tool_name not in seen:
                    seen.add(tool_name)
                    tool = self.tool_library.get_tool(tool_name)
                    if tool and tool.level > 0:  # extract only learned tools
                        implementations.append({
                            'name': tool.name,
                            'signature': tool.signature,
                            'docstring': tool.docstring,
                            'implementation': tool.implementation,
                            'used_in': source_label  # track where it's used
                        })
        return implementations
    
    def _parse_verdict(self, response):
        """parse correctness verdict and reasoning from response formatted as per correctness validation prompt"""
        verdict_match = re.search(r'<verdict>(CORRECT|INCORRECT)</verdict>', response)
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        verdict = "INCORRECT"
        if verdict_match:
            verdict = verdict_match.group(1)
        reasoning = "No reasoning provided"
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        return verdict, reasoning
