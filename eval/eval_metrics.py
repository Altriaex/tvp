"""
evaluate TVP performance metrics on Omni3D-Bench.
"""
import argparse
import json
import re
from pathlib import Path
import csv

FLOAT_ACC_TOLERANCE = 0.1
MRA_THRESHOLDS = [0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

def fuzzy_match(pred_str, gt_str, question, typ):
    """fuzzy string matching for yes/no and multiple choice answers."""
    if typ == "yes_no":
        first_word = pred_str.split()[0].strip('.,!?;:*').lower() if pred_str.split() else ""
        if gt_str == "yes":
            if first_word == "yes": # accept answers that start with "yes" as the first word
                return True
            return False
        elif gt_str == "no":
            if first_word == "no":  # accept answers that start with "no" as the first word
                return True
            if "will not" in pred_str.lower() or "would not" in pred_str.lower():   # accept "will not" or "would not" as no
                return True
            return False
    elif typ == "multiple_choice":
        for prefix in ['the ', 'a ', 'an ', 'leftmost ', 'rightmost ']:
            pred_str = pred_str.removeprefix(prefix)
            gt_str = gt_str.removeprefix(prefix)
            pred_str = pred_str.removeprefix(prefix)
            gt_str = gt_str.removeprefix(prefix)
        if pred_str == gt_str:
            return True
        if "what time is it" in question.lower().strip():   # time matching
            try:
                pred_hour, pred_min = (int(_) for _ in pred_str.split(":"))
                gt_hour, gt_min = (int(_) for _ in gt_str.split(":"))
                if gt_hour == pred_hour and gt_min == pred_min:
                    return True
            except:
                pass
        return False
    return False

class MetricEvaluator:
    def __init__(self):
        self.mra_thresholds = MRA_THRESHOLDS
        self.float_acc_tolerance = FLOAT_ACC_TOLERANCE
    
    def categorize_question(self, answer_type, ground_truth):
        if answer_type == "int":
            return "numeric_count"
        elif answer_type == "str":
            if str(ground_truth).lower() in {"yes", "no"}:
                return "yes_no"
            else:
                return "multiple_choice"
        elif answer_type == "float":
            return "numeric_other"
        else:
            return "unknown"
    
    def normalize_answer(self, answer, typ):
        if typ == "numeric_count":
            try:
                return int(round(float(str(answer).strip())))
            except:
                match = re.search(r'-?\d+\.?\d*', str(answer).strip())
                return int(round(float(match.group()))) if match else answer
        elif typ == "numeric_other":
            try:
                return float(answer)
            except:
                match = re.search(r'-?\d+\.?\d*', str(answer).strip())
                return float(match.group()) if match else answer
        elif typ == "yes_no":
            if type(answer)==bool:
                return "yes" if answer else "no"
            else:
                try:
                    answer_text = str(answer).lower().strip().strip('.,!?;:')
                    if answer_text == "true":
                        return "yes"
                    elif answer_text == "false":
                        return "no"
                    else:
                        return answer_text
                except:
                    return answer   
        elif typ == "multiple_choice":
            try:
                return str(answer).lower().strip().strip('.,!?;:')
            except:
                return answer
        return answer

    def check_answer_match(self, pred, gt, typ, question):
        pred_norm = self.normalize_answer(pred, typ)
        gt_norm = self.normalize_answer(gt, typ)
        is_correct = False
        if typ == "yes_no":
            pred_str = str(pred_norm).lower().strip()
            gt_str = str(gt_norm).lower().strip()
            is_correct = fuzzy_match(pred_str, gt_str, question, typ)
        elif typ == "multiple_choice":
            pred_str = str(pred_norm).lower().strip()
            gt_str = str(gt_norm).lower().strip()
            is_correct = fuzzy_match(pred_str, gt_str, question, typ)
        elif typ == "numeric_count":
            is_correct = pred_norm == gt_norm
        elif typ == "numeric_other": # for float, use FLOAT_ACC_TOLERANCE threshold for "correct"
            if type(pred_norm) != float:
                is_correct = False
            else:
                relative_error = abs(pred_norm - gt_norm) / abs(gt_norm)
                is_correct = relative_error < self.float_acc_tolerance
        return is_correct

    def calculate_mra_score(self, pred, gt, typ):
        pred_norm = self.normalize_answer(pred, typ)
        gt_norm = self.normalize_answer(gt, typ)
        if type(pred_norm) != float:
            return 0.0, {t: False for t in self.mra_thresholds}
        relative_error = abs(pred_norm - gt_norm) / abs(gt_norm)
        threshold_results = {}
        correct_count = 0
        for threshold in self.mra_thresholds:
            is_correct = relative_error < threshold
            threshold_results[threshold] = is_correct
            if is_correct:
                correct_count += 1
        mra = correct_count / len(self.mra_thresholds)
        return mra, threshold_results
    
    def evaluate_iteration(self, results, iteration):
        metrics = {
            "total_questions": 0,
            "execution_failures": 0,
            "categories": {
                "yes_no": {"total": 0, "correct": 0, "failed": 0},
                "multiple_choice": {"total": 0, "correct": 0, "failed": 0},
                "numeric_count": {"total": 0, "correct": 0, "failed": 0},
                "numeric_other": {"total": 0, "correct": 0, "failed": 0, "mra_sum": 0.0},
            },
            "mra_threshold_correct": {t: 0 for t in self.mra_thresholds},
            "per_question_results": []
        }
        for result in results:
            question_idx = result["question_index"]
            gt = result["answer"]
            answer_type = result["answer_type"]
            typ = self.categorize_question(answer_type, gt)
            pred = result["pred_answer"].get(iteration)
            metrics["total_questions"] += 1
            metrics["categories"][typ]["total"] += 1
            if pred is None:
                metrics["execution_failures"] += 1
                metrics["categories"][typ]["failed"] += 1
                metrics["per_question_results"].append({
                    "question_index": question_idx,
                    "category": typ,
                    "ground_truth": gt,
                    "prediction": None,
                    "correct": False,
                    "execution_failed": True
                })
            else:
                is_correct = self.check_answer_match(pred, gt, typ, result['question'])

                if is_correct:
                    metrics["categories"][typ]["correct"] += 1
                metrics["per_question_results"].append({
                    "question_index": question_idx,
                    "category": typ,
                    "ground_truth": gt,
                    "prediction": pred,
                    "correct": is_correct,
                    "execution_failed": False
                })
                if typ == "numeric_other":
                    mra_score, threshold_results = self.calculate_mra_score(pred, gt, typ)
                    metrics["categories"][typ]["mra_sum"] += mra_score
                    for threshold, is_correct_at_threshold in threshold_results.items():
                        if is_correct_at_threshold:
                            metrics["mra_threshold_correct"][threshold] += 1
        accuracies = {}
        for typ in ["yes_no", "multiple_choice", "numeric_count", "numeric_other"]:
            cat_data = metrics["categories"][typ]
            accuracies[typ] = cat_data["correct"] / cat_data["total"]
        total_exact_match = sum(
            metrics["categories"][typ]["correct"]
            for typ in ["yes_no", "multiple_choice", "numeric_count", "numeric_other"]
        )
        total_exact_match_questions = sum(
            metrics["categories"][typ]["total"] 
            for typ in ["yes_no", "multiple_choice", "numeric_count", "numeric_other"]
        )
        accuracies["overall_exact_match"] = total_exact_match / total_exact_match_questions
        num_other_data = metrics["categories"]["numeric_other"]
        accuracies["numeric_other_mra"] = num_other_data["mra_sum"] / num_other_data["total"]
        metrics["accuracies"] = accuracies
        return metrics
    
    def evaluate_all_iterations(self, data):
        results = data["results"]
        summary = data["summary"]
        iterations = set()
        for result in results:
            iterations.update(result["pred_answer"].keys())
        iterations = sorted(list(iterations))
        evaluation = {
            "summary": summary,
            "iterations": iterations,
            "metrics_by_iteration": {},
            "comparison": {}
        }
        for iteration in iterations:
            evaluation["metrics_by_iteration"][iteration] = self.evaluate_iteration(results, iteration)
        comparison = {
            "execution_success_rate": {},
            "accuracies": {
                "yes_no": {},
                "multiple_choice": {},
                "numeric_count": {},
                "numeric_other": {},
                "numeric_other_mra": {},
                "overall_exact_match": {},
            }
        }
        for iteration in iterations:
            metrics = evaluation["metrics_by_iteration"][iteration]
            total = metrics["total_questions"]
            failures = metrics["execution_failures"]
            comparison["execution_success_rate"][iteration] = (total - failures) / total if total > 0 else 0
            for typ, value in metrics["accuracies"].items():
                comparison["accuracies"][typ][iteration] = value
        evaluation["comparison"] = comparison
        return evaluation

def print_results(evaluation):
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    summary = evaluation["summary"]
    print(f"\nRun ID: {summary['run_id']}")
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Questions in Library: {summary['questions_in_library']}")
    print(f"Learned Tools: {summary['learned_tools']}")
    print(f"Iterations: {summary['iterations']}")
    print("\n" + "-"*80)
    print("PERFORMANCE ACROSS ITERATIONS")
    print("-"*80)
    iterations = evaluation["iterations"]
    comparison = evaluation["comparison"]
    print("\nExecution Success Rate:")
    for iter_name in iterations:
        rate = comparison["execution_success_rate"][iter_name]
        print(f"  {iter_name}: {rate:.2%}")
    print("\nAccuracies by Question Type:")
    for typ, iter_data in comparison["accuracies"].items():
        print(f"\n{typ.title()}:")
        for iter_name in iterations:
            value = iter_data.get(iter_name)
            if value is not None:
                print(f"  {iter_name}: {value:.2%}")
            else:
                print(f"  {iter_name}: N/A")
    print("\n" + "-"*80)
    print("DETAILED METRICS BY ITERATION")
    print("-"*80)
    for iter_name in iterations:
        print(f"\n{iter_name.upper()}:")
        metrics = evaluation["metrics_by_iteration"][iter_name]
        print(f"  Total Questions: {metrics['total_questions']}")
        print(f"  Execution Failures: {metrics['execution_failures']}")
        print("\n  Category Breakdown:")
        for typ, data in metrics["categories"].items():
            print(f"    {typ}:")
            print(f"      Total: {data['total']}")
            print(f"      Failed: {data['failed']}")
            print(f"      Correct: {data['correct']}")

def save_results_csv(evaluation, output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for iter_name in evaluation["iterations"]:
        metrics = evaluation["metrics_by_iteration"][iter_name]
        csv_path = output_path / f"{iter_name}_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question_index", "category", "ground_truth", 
                "prediction", "correct", "mra_score", "execution_failed"
            ])
            writer.writeheader()
            writer.writerows(metrics["per_question_results"])
    summary_data = {
        "summary": evaluation["summary"],
        "metrics_performance": evaluation["comparison"]
    }
    summary_path = output_path / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate metrics on Omni3D-Bench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-id", type=str, required=True,
                        help="Run ID to evaluate")
    parser.add_argument("--results-path", type=str, default="./eval/extracted_results",
                        help="Path to extracted results directory")
    parser.add_argument("--output-path", type=str, default="./eval/metrics",
                        help="Path to output directory")
    args = parser.parse_args()
    
    results_file = Path(args.results_path) / f"{args.run_id}.json"
    with open(results_file) as f:
        data = json.load(f)
    
    evaluator = MetricEvaluator()
    evaluation = evaluator.evaluate_all_iterations(data)
    print_results(evaluation)
    
    output_dir = Path(args.output_path) / args.run_id
    save_results_csv(evaluation, output_dir)
