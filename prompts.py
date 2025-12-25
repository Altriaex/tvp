"""
all prompt templates
"""
import re

from utils import extract_given_tools_from_code

def program_generation_prompt(question, examples, tools):
    basic_tools = [t for t in tools if t['level'] == 0]
    learned_tools = [t for t in tools if t['level'] > 0]
    sections = []
    # task
    sections.append(f"""You are an expert logician capable of answering spatial reasoning problems with code. You excel at using predefined APIs to write a program that answers visual-spatial and complex reasoning problems.

=== YOUR TASK ===

Question: {question}

Write a program using the provided APIs to answer this question. The variable `image` is already initialized - do not modify it!

**Requirements**:
- Store the final answer in 'final_result'
- Do NOT round your answers - leave all answers as decimals/floats
- Use existing higher-level tools when available rather than reimplementing functionality
- Learn from the example patterns but adapt to your specific question""")
    # 3D spatial concepts
    sections.append("""=== CRITICAL CONCEPTS ===

**Spatial Definitions**:
- **Axes**: Width = x-axis, Height = y-axis, Length = z-axis
- **Depth**: Distance from camera (smaller depth = closer to camera)
- **2D measurements**: Size/distance in pixel space (image coordinates)
- **3D measurements**: Size/distance in real world
- **3D size**: Formula: `3D size = 2D size × depth`
- **2D distance**: Euclidean distance between object center coordinates: `((x1-x2)**2 + (y1-y2)**2)**0.5`
- **3D distance**: Combines 2D distance with depth difference. Formula: `3D_distance = (2D_distance**2 + (depth1 - depth2)**2)**0.5`
- **Distance to camera**: Simply the object's depth value (camera is not a locatable object in the image)

**Key Principles**:
1. If two objects have the same 2D width/height, the one with larger depth has larger 3D width/height
2. To convert 2D size/distance to 3D size/distance, apply the formula defined above
3. "How many X to get to Y" means division - do not round or ceiling
4. When comparing objects, remove reference objects from search results using `same_object`
5. "Leftmost"/"rightmost" determined by object CENTER coordinates, not edges
6. To check containment: `vqa(image, 'Is this in the [container]?', object_bbox)`
7. The `loc()` function does not handle compound descriptions with properties or relationships. To find objects with specific properties, locate ALL instances of the base object first, then programmatically filter for the desired condition (e.g., use `loc(image, 'cabinet')` then filter for position, NEVER `loc(image, 'cabinet above stovetop')`)
8. Always check ALL objects satisfying a condition, not just the first one. Handle ties properly when multiple objects share the same property value
9. Hypothesis measurements (e.g., "X is 1.5m wide") should be used ONLY to scale outputs
10. When the question assumes hypothetical number of objects (e.g., "if a table has X legs..."), programmatically count the actual tables in the image - do not just use the given numbers
11. Variable "image" is already initialized - do not modify it!""")
    # available tools
    tool_counts = f"{len(basic_tools)} basic tools and {len(learned_tools)} learned (higher-level) functions" if learned_tools else f"{len(tools)} tools"
    tools_section = f"""=== AVAILABLE TOOLS ===

You have {len(tools)} tools available: {tool_counts}.

**Tool Usage Guidelines**:
1. **Use the Right Tools**: Choose tools that match your task requirements
2. **Avoid Reimplementation**: If a higher-level tool provides the functionality you need, use it rather than reimplementing with low-level tools
3. **Combine as Needed**: Complex tasks may require combining multiple tools
4. **No Synonyms**: Do not use synonyms, even if they are present in the question

"""
    if basic_tools:
        tools_section += "**BASIC TOOLS**:\n\n"
        for tool in basic_tools:
            tools_section += f"{tool['signature']} [BASIC]\n{tool['docstring']}\n\n"
    if learned_tools:
        tools_section += "**LEARNED FUNCTIONS [Level 1+]** - Use these to simplify your code:\n\n"
        for tool in learned_tools:
            tools_section += f"{tool['signature']} [Level {tool['level']}]\n{tool['docstring']}\n\n"
    sections.append(tools_section.rstrip())
    # retrieved examples
    if examples:
        example_texts = []
        for i, ex in enumerate(examples, 1):
            quality = f" (Quality: {ex['quality_rating']}/10)" if 'quality_rating' in ex else ""
            tools_list = extract_given_tools_from_code(ex['program'], tools)
            tools_line = f"\nTools used: {', '.join(sorted(set(tools_list)))}" if tools_list else ""
            example_texts.append(f"""**Example {i}**{quality}
Question: {ex['question']}{tools_line}

```python
{ex['program']}
```""")
        sections.append("=== EXAMPLE IMPLEMENTATIONS ===\n\n" + '\n\n'.join(example_texts) + "\n\nNote: Learn from these patterns but adapt to your specific question.")
    # output format
    sections.append("""=== REQUIRED OUTPUT FORMAT ===

Your response MUST be in exactly this format:

<program>
# Store the final answer in 'final_result'
final_result = your_answer
</program>

**IMPORTANT**: 
- Inside the <program> tags, include ONLY valid Python code
- No markdown formatting (NO ```python```)
- No explanations outside of code comments
- The code must be directly executable
- Show your reasoning as comments in the code""")
    return '\n\n'.join(sections)

def quality_judge_prompt(question, program, execution_result, execution_error, tools):
    # clean code --> remove markdown if present
    clean_code = program
    clean_code = re.sub(r'```python\s*\n', '', clean_code)  
    clean_code = re.sub(r'```\s*$', '', clean_code)
    clean_code = re.sub(r'```', '', clean_code)
    basic_tool_list = [t for t in tools if t['level'] == 0]
    learned_tool_list = [t for t in tools if t['level'] > 0]
    learned_tool_count = len(learned_tool_list)
    tools_used = extract_given_tools_from_code(clean_code, tools)
    # execution status and full results
    if execution_error:
        exec_status = "FAILED"
        answer_text = "None"
        execution_namespace_text = "N/A (execution failed)"
    else:
        exec_status = "SUCCESS"
        if isinstance(execution_result, dict):
            answer = execution_result.get('final_result', 'None')
            execution_namespace_text = "```\n"
            for key, value in execution_result.items():
                execution_namespace_text += f"{key}: {value}\n"
            execution_namespace_text += "```"
        else:
            answer = 'None'
            execution_namespace_text = "N/A"
        answer_text = str(answer)
    # tools documentation
    basic_tools_docs = ""
    for tool in basic_tool_list:
        basic_tools_docs += f"**{tool['name']}**\n"
        basic_tools_docs += f"{tool['signature']}\n"
        basic_tools_docs += f"{tool['docstring']}\n\n"
    learned_tools_docs = ""
    for tool in learned_tool_list:
        learned_tools_docs += f"**{tool['name']}**\n"
        learned_tools_docs += f"{tool['signature']}\n"
        learned_tools_docs += f"{tool['docstring']}\n\n"
    prompt = f"""You are an expert judge evaluating the quality of a program that solves a 3D spatial reasoning problem with tools (functions). Your task is to assess the program based on specific criteria and assign a quality rating from 1.0 to 10.0.

## TASK OVERVIEW

### Question
{question}

### Program to Evaluate

```python
{clean_code}
```

### Execution Results
- **Status:** {exec_status}
- **Final Answer:** {answer_text}
- **Tools Used:** {', '.join(tools_used) if tools_used else 'None'}
- **Execution Error:** {execution_error if execution_error else 'None'}

### Execution Namespace (All Variables)
{execution_namespace_text}

## EVALUATION FRAMEWORK

### Visual Evidence Verification
You are provided with an image of the scene. Use this visual evidence throughout your evaluation to:
- Verify the program's approach aligns with what's visible in the image
- Validate the answer's reasonableness based on visual proportions
- Check if 3D spatial relationships are correctly interpreted
- Confirm intermediate results match the visual scene
- Ensure all verification considers 3D content, not just 2D positioning

### Namespace Analysis Considerations
When reviewing the execution namespace, specifically check for:
- Correct object identification (right object selected from multiple candidates)
- Proper bounding box matching
- Expected intermediate calculations
- Appropriate object filtering
- Correct depth-based 3D conversions
- Mismatched bounding boxes or coordinates
- Unexpected intermediate calculation results
- Objects that should have been filtered but weren't

### 3D Spatial Concepts & Definitions

**Core Definitions:**
- **Coordinate system:** (width, height, length) = (x, y, z) axis
- **Depth:** Distance from camera (smaller depth = closer to camera)
- **2D measurements:** Size/distance in pixel space (image coordinates)
- **3D measurements:** Size/distance in real world
- **3D size formula:** `3D size = 2D size × depth`
- **2D distance formula:** Euclidean distance between object center coordinates: `((x1-x2)**2 + (y1-y2)**2)**0.5`
- **3D distance formula:** `3D_distance = (2D_distance**2 + (depth1 - depth2)**2)**0.5`
- **Distance to camera:** Simply the object's depth value

**Key Considerations:**
- 2D sizes are in pixel space. To convert to 3D size, multiply by depth
- Objects with same 2D dimensions but different depths have different 3D sizes
- 3D distance requires the Pythagorean formula combining 2D distance and depth difference - as defined above
- Center coordinates should determine "leftmost"/"rightmost"
- The `loc()` function should not handle compound descriptions - must locate base objects then filter for the desired condition
- All objects satisfying a condition must be checked, not just the first
- Multiple objects with same property values require proper tie-breaking
- Hypothetical object counts (e.g., "if a table has X legs") require counting actual objects in image

## RATING CRITERIA (1.0 - 10.0 Scale)

### 1. **3D Spatial Understanding**
- Properly converts between 2D and 3D measurements
- Correctly handles 3D size/distance calculation
- Correctly uses center coordinates for distance calculations and leftmost/rightmost determinations
- Interprets spatial relationships correctly (e.g., "largest" means 3D, not 2D)
- Answer is visually verifiable and reasonable

### 2. **Correctness and Visual Verification**
- Solves the problem correctly based on the actual image
- Aligns with visual evidence from the image
- Intermediate results are consistent with visible scene
- Spatial relationships match visual reality

### 3. **Tool Usage Efficiency**
- Uses appropriate tools for the task
- Leverages higher-level "learned" tools when suitable
- Avoids reimplementing existing functionality
- Note: Basic tools are acceptable when no higher-level tools fit

### 4. **Code Quality**
- Well-structured with clear variable names
- Follows tool usage patterns correctly
- Efficient without unnecessary operations
- Includes helpful comments

### 5. **Robustness and Edge Cases**
- Properly filters located objects for properties rather than using complex `loc()` queries
- Handles multiple objects with same property (proper tie-breaking)
- Manages empty lists and None values appropriately
- Manages container relationships (e.g., "in", "on") properly 
- Includes appropriate error checking

## REQUIRED OUTPUT FORMAT

You MUST provide your response in exactly this format:

<rating>NUMBER</rating>
<reasoning>
[Detailed explanation covering:
- How visual evidence supports/contradicts the program's logic
- Specific strengths and weaknesses identified
- Analysis of 3D spatial reasoning approach
- Evaluation of intermediate execution results
- Missed opportunities to use available tools
- Overall assessment based on all criteria]
</reasoning>

Where NUMBER is a decimal between 1.0 and 10.0.

---

## APPENDIX: Available Tools Reference

The program had access to {len(tools)} tools total: {len(basic_tool_list)} basic tools and {learned_tool_count} learned tools.

### Basic Tools (Level 0)
{basic_tools_docs}
### Learned Tools (Level 1+)
{learned_tools_docs}"""
    return prompt

def cluster_analysis_prompt(examples):
    # format examples to analyze
    examples_text = ""
    example_ids = []
    for i, ex in enumerate(examples):
        example_ids.append(ex['id'])
        examples_text += f"\n### Example {i+1} (ID: {ex['id']})\n"
        examples_text += f"**Question:** {ex['question']}\n"
        if ex.get('uses_tools'):
            examples_text += f"\n**Uses Tools:** {', '.join(ex['uses_tools'])}\n"
        examples_text += f"\n**Program:**\n```python\n{ex['program']}\n```\n"
    ids_list = ", ".join(example_ids)   # list of all example ids for the final reminder
    prompt = f"""You are an expert at analyzing visual reasoning programs to identify common patterns that could be abstracted into reusable functions.

## Your Task

Analyze {len(examples)} visual reasoning examples to:
1. Identify common computational patterns across examples
2. Group them into clusters based on shared functionality  
3. Rate each cluster's abstraction potential (0-10 scale)

## Examples to Analyze
{examples_text}

## Clustering Criteria

Identify clusters based on:

1. **Common computational patterns** - e.g., finding largest/smallest, counting with conditions
2. **Similar operations sequence** - e.g., locate -> filter -> compute -> compare
3. **Shared logic structure** - e.g., iteration patterns, comparison logic
4. **Abstractable functionality** - can be parameterized into a reusable function

## Evaluation Requirements

### For Each Cluster Provide:
- **Example IDs** that belong to it
- **Common pattern** explanation
- **Parameters** that vary between examples
- **Abstraction potential rating** (0-10) based on:
  * How well the pattern generalizes
  * Parameter variability coverage
  * Clarity of the abstraction
  * Reusability across similar tasks
- **Reasoning** for the rating

### Critical Constraints
- **Each example must belong to exactly ONE cluster or be marked as unclustered**
- Focus on computational patterns, not surface similarities
- Only create clusters with strong shared patterns

## Response Format

Provide your analysis using this exact format. Include as many cluster blocks as needed, followed by an optional unclustered block:

```
<cluster>
<example_ids>[comma-separated list of example IDs]</example_ids>
<pattern>[Description of the common computational pattern]</pattern>
<parameters>[List of parameters that vary between examples]</parameters>
<abstraction_potential>[Rating from 0-10]</abstraction_potential>
<reasoning>[Explanation for the rating and how the pattern could be abstracted]</reasoning>
</cluster>

[Additional <cluster> blocks as needed...]

<unclustered>
<example_ids>[comma-separated list of example IDs that don't fit clusters]</example_ids>
<reasoning>[Explanation of why these examples don't cluster well]</reasoning>
</unclustered>
```

**Remember:** Every example ID ({ids_list}) must appear in exactly ONE cluster or in the unclustered group."""
    return prompt

def tool_abstraction_prompt(cluster_info, examples, existing_tools, used_tool_names=None):
    # examples to abstract
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += f"### Example {i}\n"
        examples_text += f"**Question**: {ex['question']}\n"
        examples_text += f"**Program**:\n"
        examples_text += f"```python\n{ex['program']}\n```\n\n"
    # forbidden func names (already used in tool lib)
    forbidden_names = ', '.join([f"`{name}`" for name in used_tool_names]) if used_tool_names else '(none yet)'
    # existing tools for reference section
    available_tools_text = ""
    for tool in existing_tools:
        available_tools_text += f"**{tool['name']}**:\n"
        available_tools_text += f"{tool['signature']}\n"
        available_tools_text += f"{tool['docstring']}\n\n"
    prompt = f"""You are an expert at abstracting reusable functions from clusters of similar 3D spatial reasoning programs.

## Your Task

Abstract the following cluster of programs into a single reusable function.

### Identified Pattern
**Common Pattern**: {cluster_info['pattern']}

**Parameters that vary**: {cluster_info['parameters']}

**Abstraction Potential**: {cluster_info['abstraction_potential']}/10 - {cluster_info.get('reasoning', 'Pattern shows strong potential for abstraction')}

## Example Programs to Abstract

{examples_text}## Implementation Requirements

### Core Requirements
1. **Preserve 3D Logic**: Maintain depth handling and 3D calculations exactly as shown in examples
2. **Parameter Design**: 
   - First parameter must be `image`
   - Parameters should emerge naturally from the pattern identified above
   - Support variations seen in examples: {cluster_info['parameters']}
3. **No Assumptions**: Don't assume units or specific objects - keep function general
4. **Edge Cases**: Handle empty lists, None values, missing objects appropriately

### Critical Restrictions
- **NO callable parameters**: Do NOT accept functions, lambdas, or callables as parameters
- **NO rounding**: Maintain full precision in calculations
- **Function name**: Must start with underscore (e.g., `_find_largest_by_3d_size`)
- **Forbidden names**: Do NOT use any of these function names:
  {forbidden_names}

### Documentation Standards
- Include comprehensive docstring with Args and Returns sections
- Explain 3D aspects clearly
- Document what "largest", "closest", etc. mean in 3D context
- Include comments in implementation explaining 3D conversions

## Output Format

Provide your abstracted function in the following format:

```
<function_name>_descriptive_function_name</function_name>
<signature>def _descriptive_function_name(image, [additional parameters as needed]):</signature>
<docstring>\"\"\"
Comprehensive description of what the function does, especially any 3D calculations.

Args:
    image: Input image to analyze
    [parameter_name]: [Type expected] Description of what this parameter does
    [... additional parameters as needed based on the pattern]
    
Returns:
    [Type]: Description of what the function returns
\"\"\"</docstring>
<implementation>
# Your implementation here (properly indented, no function signature)
# Preserve 3D logic exactly as shown in examples
# Include comments explaining 3D conversions
objects = loc(image, prompt)  # Example
result = some_computation()
return result
</implementation>
<abstraction_reasoning>Explain how you implemented the identified pattern, especially how you preserved the 3D spatial reasoning logic and handled parameter variations</abstraction_reasoning>
```

**IMPORTANT**: The `<implementation>` tag should contain ONLY the function body:
- Do NOT include the function signature (`def ...:`)
- Do NOT include the docstring
- Start with properly indented code logic
- Use any available tools from the reference section below

---

## Available Tools Reference

You may use any of these existing functions in your implementation:

{available_tools_text}"""
    return prompt

def program_rewrite_prompt(original_program, new_tool, question):
    prompt = f"""Rewrite the original program to use the newly created `{new_tool['name']}` function.

## Output Requirements
**Provide ONLY the rewritten Python code** - no additional text, markdown formatting, or explanations. The code should be directly executable and store the result in `final_result`.

## Context

### Original Question
{question}

### Original Program
```python
{original_program}
```

### New Function Available
```python
{new_tool['signature']}
{new_tool['docstring']}
```

## Rewriting Guidelines

### Core Requirements
1. **Use the new function appropriately** - Replace the old program's implementation with a call to `{new_tool['name']}`
2. **Maintain the same logic** - The rewritten program must produce the same answer as the original
3. **Store result correctly** - Final result must be stored in the variable `final_result`
4. **Preserve numerical precision** - Do not round any numerical results

### Implementation Notes
- **Only replace functionality covered by the new function** - Keep any question-specific logic that isn't part of the abstracted pattern
- **Match parameter types carefully** - Ensure the parameters you pass match what the function expects
- **Use optional parameters when needed** - Check if filter_question, select_shortest, or other optional parameters apply to your use case
- **Preserve any necessary setup or post-processing** - Keep code that's specific to this question but not handled by the function
- **Maintain unit consistency** - Ensure any unit conversions or scaling from the original are preserved"""
    return prompt

def tool_deduplication_prompt(tools):
    tools_text = ""
    for i, tool in enumerate(tools):
        tools_text += f"\n**TOOL {i+1}: {tool['name']}**\n"
        tools_text += f"{tool['signature']}\n"
        tools_text += f"{tool['docstring']}\n"
        tools_text += f"Implementation:\n```python\n{tool['implementation']}\n```\n"
    prompt = f"""You are analyzing a set of tools to identify duplicates or highly similar functions that should be merged to reduce redundancy while maintaining usability.

## Tools to Analyze
{tools_text}

## Evaluation Criteria

Identify groups of tools that meet ALL of the following criteria:

1. **Core Functionality Similarity**: Tools perform fundamentally the same type of operation (e.g., both compute ratios, both find extrema, both aggregate measurements)

2. **Parameter Overlap**: Tools have either:
   - Identical parameter structures with different names, OR
   - One tool's parameters are a subset/superset of another's

3. **Algorithm Similarity**: The underlying logic and computational steps are similar enough that they could share implementation

4. **Use Case Compatibility**: The merged tool would serve all original use cases without becoming unwieldy or confusing

## Merging Constraints

- **Uniqueness Rule**: Each tool index can appear in ONLY ONE merge group. When a tool could potentially fit multiple groups, assign it to the group with the highest similarity score.

- **Usability Requirement**: Only propose merges that maintain simplicity. The merged tool should:
  - Have clear, intuitive parameters
  - Not require complex configuration flags
  - Be as easy to maintain as separate tools

- **Naming Convention**: Merged function names must start with underscore (_) and should NOT include version suffixes like "_v2", "_v3", etc.

- **No Forced Merges**: If no tools should be merged, explicitly state this rather than forcing low-quality merges

## Output Format

For each identified group of similar tools, provide your analysis in the following format:

<merge_group>
<tool_indices>List indices separated by commas (e.g., 1, 3, 5)</tool_indices>
<similarity_score>Float between 0.0 and 1.0 (including decimal points)</similarity_score>
<merge_strategy>Explain how to combine the tools to handle all original cases</merge_strategy>
<recommended_name>Suggested merged function name (must start with _)</recommended_name>
<reasoning>Detailed explanation of why these tools should be merged, their similarities, and the proposed unification approach</reasoning>
</merge_group>

Separate each merge group with double newlines."""
    return prompt

def tool_merge_implementation_prompt(tools, merge_strategy, merge_strategy_reason, existing_tools, used_tool_names=None):
    # tools to merge section
    tools_text = ""
    for i, tool in enumerate(tools, 1):
        tools_text += f"\n**Tool {i}: {tool['name']}**\n"
        tools_text += f"```python\n"
        tools_text += f"{tool['signature']}\n"
        tools_text += f"{tool['docstring']}\n"
        tools_text += f"# Implementation:\n"
        tools_text += f"{tool['implementation']}\n"
        tools_text += f"```\n"
    # format existing tools for use in implementation
    available_tools_text = ""
    for tool in existing_tools:
        if any(t['name'] == tool['name'] for t in tools):   # skip the tools to merge
            continue
        available_tools_text += f"**{tool['name']}**:\n"
        available_tools_text += f"{tool['signature']}\n"
        available_tools_text += f"{tool['docstring']}\n\n"
    # forbidden func names (already used in tool lib)
    forbidden_names = ', '.join(used_tool_names) if used_tool_names else '(none specified)'
    # merge analysis from tool dedup
    merge_analysis = f"{merge_strategy}\n\n{merge_strategy_reason}"
    prompt = f"""Create a unified function that elegantly combines the functionality of similar tools listed below, eliminating redundancy while preserving all capabilities.

## Tools to Merge
{tools_text}
## Merge Analysis
{merge_analysis}

## Requirements for Merged Function

### Core Functionality
- **Preserve all capabilities**: The merged function must handle all use cases from both original tools
- **Unified interface**: Create clear, intuitive parameters that cover all scenarios
- **Comprehensive documentation**: Write a detailed docstring describing all functionalities (do not simply state "combining capabilities of...")

### Design Constraints
- **No callable parameters**: Do NOT accept functions, lambdas, or callables as parameters
- **First parameter**: Must be `image` (consistent with all visual functions)
- **Forbidden function names**: Do NOT use any of these names: {forbidden_names}
- **User-friendly design**: The merged function should be intuitive and not overly complicated

### Implementation Guidelines
- Use the merge strategy as a reference but adapt as needed for optimal implementation
- Only use functions from the Available Tools list (see below)
- Design parameters that naturally emerge from merge requirements
- Maintain clear error handling and validation
- Use only available callable tools: You can use any of the available tools (functions) shown below in your implementation, but not any non-existent tool calls that are not provided to you

## Available Tools for Implementation
{available_tools_text}
## Expected Output Format

Provide your response in this exact format:

```
<function_name>_descriptive_merged_name</function_name>
<signature>def _descriptive_merged_name(image, [additional parameters as needed]):</signature>
<docstring>\"\"\"
Detailed description covering all functionalities from all merged functions

Args:
    image: Input image to analyze
    [parameter_name]: [Type expected] Description covering all merged use cases
    [... additional parameters as needed]
    
Returns:
    [Type]: Description of return value
\"\"\"</docstring>
<implementation>
# Unified implementation (no initial indentation)
# Combine the best aspects of all merged tools
# DO NOT include function signature or docstring here
# Start directly with the code logic
</implementation>
```

### Formatting Requirements
- `<function_name>` contains ONLY the function name (no "_v2", "_v3", etc.)
- `<signature>` contains the complete function signature
- `<docstring>` contains the triple-quoted documentation string
- `<implementation>` contains ONLY the function body code (no signature, no docstring)"""
    return prompt

def correctness_validation_prompt(question, original_answer, new_answer, original_program, rewritten_program,  tool_implementations=None, original_namespace=None, new_namespace=None):
    # documentation of tools used
    tool_section = ""
    if tool_implementations:
        original_tools = [t for t in tool_implementations if t.get('used_in') in ['original', 'both']]
        rewritten_tools = [t for t in tool_implementations if t.get('used_in') in ['rewritten', 'both']]
        if original_tools:
            tool_section += "**Tools used in ORIGINAL program:**\n"
            for tool in original_tools:
                tool_section += f"\n`{tool['name']}`:\n"
                tool_section += f"{tool['signature']}\n"
                tool_section += f"{tool['docstring']}\n"
        if rewritten_tools:
            tool_section += "**Tools used in REWRITTEN program:**\n"
            for tool in rewritten_tools:
                if any(t['name'] == tool['name'] for t in original_tools):  # skip if already shown in original tools
                    tool_section += f"\n`{tool['name']}`: (same as in ORIGINAL program)\n"
                else:
                    tool_section += f"\n`{tool['name']}`:\n"
                    tool_section += f"{tool['signature']}\n"
                    tool_section += f"{tool['docstring']}\n"
    # format full execution results (namespaces)
    original_namespace_text = ""
    if original_namespace and isinstance(original_namespace, dict):
        for key, value in original_namespace.items():
            original_namespace_text += f"{key}: {value}\n"
    new_namespace_text = ""
    if new_namespace and isinstance(new_namespace, dict):
        for key, value in new_namespace.items():
            new_namespace_text += f"{key}: {value}\n"
    prompt = f"""You are validating whether a rewritten program produces an answer that is **AS GOOD AS OR BETTER** than the original answer. Neither answer is guaranteed to be correct - both could be right, wrong, or one could be correct while the other is wrong. **Do NOT treat the original answer as the gold standard.**

## YOUR TASK
Determine if the new answer is **ACCEPTABLE** by:
1. Analyzing both programs' logic and execution
2. Verifying answers against the actual image provided
3. Considering 3D spatial relationships, not just 2D positioning
4. Examining intermediate values in execution namespaces for insights

## 3D SPATIAL FUNDAMENTALS

### Coordinate System Definitions
- **Axes**: Width = x-axis, Height = y-axis, Length = z-axis
- **Depth**: Distance from camera (smaller depth = closer to camera)
- **2D space**: Pixel coordinates in the image plane
- **3D space**: Real-world coordinates with depth from camera
- **3D size formula**: `3D size = 2D size × depth`
- **2D distance formula**: Euclidean distance between  center coordinates: `((x1-x2)**2 + (y1-y2)**2)**0.5`
- **3D distance formula**: `3D_distance = (2D_distance**2 + (depth1 - depth2)**2)**0.5`
- **Distance to camera**: Simply the object's depth value

### Critical Spatial Concepts
- Objects with identical 2D dimensions have different 3D sizes based on depth
- 3D distance calculation requires Pythagorean formula combining 2D distance and depth difference
- "Leftmost/rightmost" determinations should use object center coordinates
- The `loc()` function should not handle compound descriptions - must locate base objects then filter
- When multiple objects share property values (e.g., same min/max), proper tie-breaking is needed

## CASE TO VALIDATE

### QUESTION
{question}

### TOOLS BEING USED
{tool_section}

### ORIGINAL PROGRAM
```python
{original_program}
```
**Original Answer**: {original_answer}
**Original Execution Namespace**: {original_namespace_text}

### REWRITTEN PROGRAM
```python
{rewritten_program}
```
**New Answer**: {new_answer}
**New Execution Namespace**: {new_namespace_text}

## VALIDATION GUIDELINES

**Remember**: We do NOT know which answer is correct. Both could be right, wrong, or one could be right while the other is wrong. Judge whether the new answer is as good as or better than the original, NOT whether it matches the original.

### Tool Analysis
When programs use different tools:
- **Different tools can be equally valid**: Tools with different names may accomplish the same task correctly
- **Focus on semantic correctness**: Verify that each tool is appropriate for the operation it performs
- **Check tool parameters**: Ensure the tools receive correct inputs (e.g., proper dimension axes, correct object references)
- **Tool abstraction is acceptable**: Simpler, more abstract tools that produce correct results should not be penalized

### Visual Evidence Verification
Using the provided image, verify:
- **Spatial Relationships**: Validate that the 3D spatial relationships of objects match visual evidence
- **Numerical Plausibility**: Check if counts/measurements are reasonable
- **3D Visual Consistency**: The answer should match visual evidence (considering the 3D sense beyond 2D pixel space)

### Namespace Analysis

**Note**: Simpler functions may produce fewer intermediate values. This transparency difference does NOT indicate inferiority.

Examine execution namespaces to identify:
- **Object selection differences**: How each program handles cases where one instance needs to be selected out of multiple objects (e.g., among objects sharing the same quality)
- **Spatial interpretation variations**: Different approaches to 3D calculations and interpreting spatial relationships
- **Potential errors**: Mismatched bounding boxes, unexpected intermediate values, or missing object filters

### Answer Acceptability Standards

- **3D Measurements**: Small differences acceptable due to depth estimation/boundary detection variations
- **Object Counts**: Must match visual evidence exactly (unless ambiguous)
- **Yes/No Answers**: Must match visual evidence exactly
- **Ratios/Percentages**: Verify both measure the same relationship
- **Detection Failures**: "Object not found" or None values indicate FAILURE (question assumes objects exist)

## REQUIRED OUTPUT FORMAT

Provide your response in exactly this format:

```
<verdict>CORRECT or INCORRECT</verdict>
<reasoning>
Provide a thorough explanation of your judgment based on the validation guidelines above. Include relevant observations about visual evidence, namespace analysis, and any critical factors that influenced your decision.
</reasoning>
```"""
    return prompt