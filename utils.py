"""
utils in TVP
"""
import re
import time
import numpy as np
import traceback
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")
from openai import OpenAI
from FlagEmbedding import FlagModel

from models import Tool
from logging_utils import get_logger

def get_predefined_tools(dataset="omni3d"):
    """get predefined tools for Omni3D dataset --> implementations in predefined_apis.py"""
    return [
        Tool(
            name="loc",
            level=0,
            signature="def loc(image, object_prompt):",
            docstring='''"""
Locates objects in an image. Object prompts should be simple and contain few words. To return all objects, pass "objects" as the object_prompt.

Args:
    image (image): Image to search.
    object_prompt (string): Description of object to locate.
Returns:
    bbox (list): A list of bounding boxes [xmin, ymin, xmax, ymax] for all of the objects located in pixel space, where (x,y) coordinates represent pixel positions in the image. xmin,ymin,xmax,ymax are the left-edge, top-edge, right-edge, bottom-edge coordinates respectively.
"""''',
            implementation="pass  # predefined",
        ),
        Tool(
            name="vqa",
            level=0,
            signature="def vqa(image, question, bbox):",
            docstring='''"""
Answers a question about an object shown in a bounding box.

Args:
    image (image): Image of the scene.
    question (string): Question about the object in the bounding box.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the object, where (x,y) coordinates represent pixel positions in the image. xmin,ymin,xmax,ymax are the left-edge, top-edge, right-edge, bottom-edge coordinates respectively.

Returns:
    string: Answer to the question about the object in the image.
"""''',
            implementation="pass  # predefined",
        ),
        Tool(
            name="depth",
            level=0,
            signature="def depth(image, bbox):",
            docstring='''"""
Returns the depth of an object (specified by a bounding box) in meters.

Args:
    image (image): Image of the scene.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the object, where (x,y) coordinates represent pixel positions in the image. xmin,ymin,xmax,ymax are the left-edge, top-edge, right-edge, bottom-edge coordinates respectively.

Returns:
    float: The depth of the object specified by the coordinates in meters.
"""''',
            implementation="pass  # predefined",
        ),
        Tool(
            name="same_object",
            level=0,
            signature="def same_object(image, bbox1, bbox2):",
            docstring='''"""
Checks if two bounding boxes correspond to the same object.

Args:
    image (image): Image of the scene.
    bbox1 (list): A bounding box [xmin, ymin, xmax, ymax] containing object1, where (x,y) coordinates represent pixel positions in the image. xmin,ymin,xmax,ymax are the left-edge, top-edge, right-edge, bottom-edge coordinates respectively.
    bbox2 (list): A bounding box [xmin, ymin, xmax, ymax] containing object2, where (x,y) coordinates represent pixel positions in the image. xmin,ymin,xmax,ymax are the left-edge, top-edge, right-edge, bottom-edge coordinates respectively.

Returns:
    bool: True if object 1 is the same object as object 2, False otherwise.
"""''',
            implementation="pass  # predefined",
        ),
        Tool(
            name="get_2D_object_size",
            level=0,
            signature="def get_2D_object_size(image, bbox):",
            docstring='''"""
Returns the width and height of the object in 2D pixel space.

Args:
    image (image): Image of the scene.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the object, where (x,y) coordinates represent pixel positions in the image. xmin,ymin,xmax,ymax are the left-edge, top-edge, right-edge, bottom-edge coordinates respectively.

Returns:
    tuple: (width, height) of the object in 2D pixel space.
"""''',
            implementation="pass  # predefined",
        )
    ]

class LLMClient:
    def __init__(self, config):
        self.config = config
        self._client = None
        self.logger = get_logger(__name__)
    
    @property
    def client(self):
        if self._client is None:
            api_key_path = Path(self.config.api_key_path)
            if not api_key_path.exists():
                raise FileNotFoundError(f"API key file not found: {api_key_path}")
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            self._client = OpenAI(api_key=api_key)
        return self._client

    def generate(self, prompt, model, temperature=1.0, reasoning_effort='medium', max_retries=3):
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(max_retries):
            try:
                # reasoning models
                if model.startswith("o"):
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        reasoning_effort=reasoning_effort
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature
                    )
                return response.choices[0].message.content
            except Exception as e:
                self.logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}), Error: {e}")
                if attempt < max_retries - 1:
                    wait_time = 0.5 * (attempt+1)
                    time.sleep(wait_time)
                else:
                    raise
    
    def generate_with_image(self, prompt, image_base64, model, temperature=1.0, reasoning_effort='medium', max_retries=3):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"}}]
            }]
        if model.startswith("o"):
            self.logger.warning(f"Model {model} does not support images, using gpt-4o instead")
            model = "gpt-4o"
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                self.logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 0.5 * (attempt+1)
                    time.sleep(wait_time)
                else:
                    raise

class EmbeddingManager:
    def __init__(self, config):
        self.config = config
        self.model_name = config.embedding_model
        self._model = None
        self.logger = get_logger(__name__)
    
    @property
    def model(self):
        if self._model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self._model = FlagModel(
                self.model_name,
                query_instruction_for_retrieval="",
                use_fp16=True
            )
        return self._model
    
    def embed_text(self, text):
        embedding = self.model.encode(text)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

class ParallelExecutor:
    def __init__(self, config):
        self.config = config
        self.max_workers = config.max_workers
        self.logger = get_logger(__name__)
    
    def execute_parallel(self, func, tasks, desc=None):
        results = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {}
            for i, task in enumerate(tasks):
                future = executor.submit(func, task)
                future_to_idx[future] = i
            futures = list(future_to_idx.keys())
            if desc:
                pbar = tqdm(total=len(futures), desc=desc)
            else:
                pbar = None
            for future in as_completed(futures):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    self.logger.error(f"Task {idx} failed: {e}")
                    self.logger.debug(f"Traceback: {traceback.format_exc()}")
                    results[idx] = None
                if pbar:
                    pbar.update(1)
            if pbar:
                pbar.close()
        return results

class ClusteringEngine:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)

    def group_by_similarity(self, examples, threshold=0.85, return_detailed_info=False):
        """group examples by question embedding similarity using greedy clustering"""
        if not examples:
            return [], None
        # compute pairwise similarity
        n = len(examples)
        embeddings = np.vstack([ex.question_embedding for ex in examples])
        similarity_matrix = embeddings @ embeddings.T
        # greedy clustering
        groups = []
        assigned = np.zeros(n, dtype=bool)
        grouping_decisions = [] if return_detailed_info else None
        rejection_reasons = [] if return_detailed_info else None
        for i in range(n):
            if assigned[i]:
                continue
            group = [i]
            assigned[i] = True
            candidates = np.where(~assigned)[0]
            candidates = candidates[candidates > i] # get unassigned candidates with index > i
            if len(candidates) > 0:
                min_sims = similarity_matrix[i, candidates].copy()  # initialize min similarity to group (group is just [i] so far)
                for idx, j in enumerate(candidates):
                    if assigned[j]:  # may have been assigned in this loop
                        continue
                    min_sim = min_sims[idx]
                    if min_sim >= threshold:
                        group.append(j)
                        assigned[j] = True
                        if idx + 1 < len(candidates):   # incrementally update min_sims for remaining candidates
                            remaining_slice = slice(idx + 1, None)
                            remaining_candidates = candidates[remaining_slice]
                            min_sims[remaining_slice] = np.minimum(
                                min_sims[remaining_slice],
                                similarity_matrix[j, remaining_candidates]
                            )
                        if return_detailed_info:
                            grouping_decisions.append({
                                'example_i': i,
                                'example_j': int(j),
                                'similarity': float(similarity_matrix[i][j]),
                                'min_similarity_to_group': float(min_sim),
                                'decision': 'grouped'
                            })
                    elif return_detailed_info:
                        rejection_reasons.append({
                            'example_i': i,
                            'example_j': int(j),
                            'similarity': float(similarity_matrix[i][j]),
                            'min_similarity_to_group': float(min_sim),
                            'reason': f'Below threshold (min_sim={min_sim:.3f} < {threshold})'
                        })
            groups.append([examples[idx] for idx in group])
        detailed_info = None 
        if return_detailed_info:    # create detailed info
            detailed_info = {
                'similarity_matrix': similarity_matrix.tolist(),
                'threshold': threshold,
                'grouping_decisions': grouping_decisions,
                'rejection_reasons': rejection_reasons,
                'group_statistics': {
                    'total_groups': len(groups),
                    'group_sizes': [len(g) for g in groups],
                    'singleton_groups': sum(1 for g in groups if len(g) == 1),
                    'average_group_size': sum(len(g) for g in groups) / len(groups) if groups else 0
                }
            }
        self.logger.info(f"Grouped {n} examples into {len(groups)} groups")
        return groups, detailed_info

def extract_tools_from_program(program):
    """extract unique tool names from a program considering predefined functions"""
    predefined = [t.name for t in get_predefined_tools()]
    predefined_pattern = r'\b(' + '|'.join(predefined) + r')\s*\('
    tool_patterns = [
        r'\b(_\w+)\s*\(',  # learned tools starting with underscore
        predefined_pattern  # predefined tools
    ]
    tools_used = []
    for pattern in tool_patterns:
        matches = re.findall(pattern, program)
        tools_used.extend(matches)
    seen = set()
    unique_tools = []
    for tool in tools_used:
        if tool not in seen:
            seen.add(tool)
            unique_tools.append(tool)
    return unique_tools

def extract_given_tools_from_code(code, tools):
    """extract tool names from a given list of tools"""
    tool_calls = re.findall(r'\b(\w+)\s*\(', code)
    return [t for t in tool_calls if any(tool['name'] == t for tool in tools)]

def extract_code(response, return_nan):
    """extract code from llm response formatted as per prompt templates"""
    match = re.search(r'<program>\s*(.*?)\s*</program>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    if return_nan:
        return None
    else:
        return response.strip()

def normalize_answer(answer):
    """normalize answer format"""
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    elif isinstance(answer, str):
        return answer.lower().strip()
    elif isinstance(answer, (int, float)):
        return round(float(answer), 3)
    return answer
