"""
extract results from TVP pipeline run into a single JSON file for evaluation.
"""
import argparse
import json
from pathlib import Path


def load_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return None

def extract_best_program_info(question_dir):
    best_program_path = question_dir / "best_program.json"
    best_program_data = load_json_safe(best_program_path)
    if not best_program_data:
        return {
            "program": None,
            "answer": None,
            "quality_rating": None,
            "execution_success": False,
            "error": "No best_program.json found"
        }
    if best_program_data.get("selected", False):    # successful candidate was selected
        return {
            "program": best_program_data.get("program"),
            "answer": best_program_data.get("answer"),
            "quality_rating": best_program_data.get("quality_rating"),
            "execution_success": True,
            "tools_used": best_program_data.get("tools_used", []),
            "candidate_index": best_program_data.get("candidate_index"),
            "added_to_library": best_program_data.get("added_to_library", False)
        }
    else:   # no successful candidate -> check for best failed attempt
        best_failed = best_program_data.get("best_failed_attempt", {})
        if best_failed:
            return {
                "program": best_failed.get("program", None),
                "answer": best_failed.get("answer"),
                "quality_rating": best_failed.get("quality_rating"),
                "execution_success": False,
                "error": best_failed.get("execution_error", "Unknown error"),
                "candidate_index": best_failed.get("candidate_index")
            }
        else:
            return {
                "program": None,
                "answer": None,
                "quality_rating": None,
                "execution_success": False,
                "error": best_program_data.get("reason", "No successful candidates")
            }

def load_example_library(results_dir):
    examples_path = results_dir / "state" / "examples.json"
    examples_data = load_json_safe(examples_path)
    if not examples_data:
        return {}
    examples_by_question = {}
    for example in examples_data:
        question_idx = example["question_index"]
        examples_by_question[question_idx] = example
    return examples_by_question

def load_tools(results_dir):
    tools_path = results_dir / "state" / "tools.json"
    tools_data = load_json_safe(tools_path)
    return tools_data if tools_data else []

def extract_iteration_data(results_dir, question_idx, num_iterations):
    iteration_data = {}
    for iter_num in range(1, num_iterations + 1):
        iter_dir = results_dir / f"iterations/iteration_{iter_num}/questions/q_{question_idx}"
        if iter_dir.exists():
            best_program_info = extract_best_program_info(iter_dir)
            iteration_data[f"iter{iter_num}"] = best_program_info
    return iteration_data


def compile_metadata(question_idx, iteration_data, example_library):
    metadata = {
        "iterations_processed": len(iteration_data),
        "execution_success_by_iteration": {},
        "quality_ratings": {},
        "in_example_library": question_idx in example_library
    }
    for iter_key, iter_info in iteration_data.items():
        metadata["execution_success_by_iteration"][iter_key] = iter_info.get("execution_success", False)
        if iter_info.get("quality_rating") is not None:
            metadata["quality_ratings"][iter_key] = iter_info["quality_rating"]
    if question_idx in example_library:
        example = example_library[question_idx]
        metadata["library_info"] = {
            "final_quality_rating": example.get("quality_rating"),
            "final_tools_used": example.get("tools_used", []),
            "abstraction_status": example.get("abstraction_status", "unknown"),
            "abstracted_tools": example.get("abstracted_tools", []),
            "update_count": len(example.get("update_history", []))
        }
        if example.get("update_history"):
            metadata["update_types"] = [u["update_type"] for u in example["update_history"]]
    return metadata

def extract_results(run_id, results_path, output_file):
    results_dir = Path(results_path) / run_id
    print(f"Loading data from {results_dir}...")
    questions_path = results_dir / "sampled_questions.json"
    questions_data = load_json_safe(questions_path)
    if not questions_data:
        print(f"Error: Could not load questions from {questions_path}")
        return
    print(f"Found {len(questions_data)} questions")
    example_library = load_example_library(results_dir)
    tools = load_tools(results_dir)
    print(f"Found {len(example_library)} examples in library")
    print(f"Found {len([t for t in tools if t['level'] > 0])} learned tools")
    print(f"Found {len([t for t in tools if t['level'] > 0 and not t['is_deprecated']])} active (not deprecated) learned tools")
    iterations_dir = results_dir / "iterations"
    num_iterations = len(list(iterations_dir.glob("iteration_*")))
    print(f"Found {num_iterations} iterations")
    results = []
    for question in questions_data:
        question_idx = question["question_index"]
        iteration_data = extract_iteration_data(results_dir, question_idx, num_iterations)
        programs = {}
        pred_answers = {}
        for iter_key, iter_info in iteration_data.items():
            programs[iter_key] = iter_info.get("program")
            pred_answers[iter_key] = iter_info.get("answer")
        metadata = compile_metadata(question_idx, iteration_data, example_library)
        metadata["iteration_details"] = {}
        for iter_key, iter_info in iteration_data.items():
            metadata["iteration_details"][iter_key] = {
                "execution_success": iter_info.get("execution_success", False),
                "quality_rating": iter_info.get("quality_rating"),
                "tools_used": iter_info.get("tools_used", []),
                "error": iter_info.get("error") if not iter_info.get("execution_success") else None
            }
        result = {
            "image_index": question["image_index"],
            "question_index": question["question_index"],
            "image_filename": question["image_filename"],
            "question": question["question"],
            "answer": question["answer"],
            "answer_type": question["answer_type"],
            "program": programs,
            "pred_answer": pred_answers,
            "meta_data": metadata
        }
        results.append(result)
    summary = {
        "run_id": run_id,
        "total_questions": len(results),
        "questions_in_library": len(example_library),
        "learned_tools": len([t for t in tools if t['level'] > 0]),
        "deprecated_tools": len([t for t in tools if t['is_deprecated']]),
        "iterations": num_iterations
    }
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "summary": summary,
        "results": results
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Summary:")
    print(f"  - Total questions: {summary['total_questions']}")
    print(f"  - Questions in library: {summary['questions_in_library']}")
    print(f"  - Learned tools: {summary['learned_tools']}")
    print(f"  - Deprecated tools: {summary['deprecated_tools']}")
    print(f"  - Iterations: {summary['iterations']}")
    all_executions = [s for r in results for s in r["meta_data"]["execution_success_by_iteration"].values()]
    total_executions, successful_executions = len(all_executions), sum(all_executions)
    if total_executions > 0:
        print(f"  - Overall execution success rate: {successful_executions}/{total_executions} "
              f"({100*successful_executions/total_executions:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract results from TVP pipeline run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-id", type=str, required=True,
                        help="Run ID to extract results from")
    parser.add_argument("--results-path", type=str, default="./results",
                        help="Path to results directory")
    parser.add_argument("--output-path", type=str, default="./eval/extracted_results",
                        help="Path to output directory")
    args = parser.parse_args()
    output_file = Path(args.output_path) / f"{args.run_id}.json"
    extract_results(args.run_id, args.results_path, output_file)
