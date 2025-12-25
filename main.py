"""
main entry point for TVP library building pipeline.
"""
import argparse
import sys
import os

from config import create_config_from_args
from pipeline import Pipeline
from logging_utils import setup_logging, log_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = argparse.ArgumentParser(
        description="TVP library building pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # dataset and paths
    parser.add_argument("--dataset", type=str, default="omni3d", choices=["omni3d"],
                       help="dataset type")
    parser.add_argument("--annotations-json", type=str, default="./data/omni3d-bench/annotations.json",
                       help="path to annotations json file")
    parser.add_argument("--image-path", type=str, default="./data/omni3d-bench/images/",
                       help="path to image directory")
    parser.add_argument("--models-path", type=str, default="./models/",
                       help="path to vision tool model directory")
    parser.add_argument("--results-path", type=str, default="./results/",
                       help="path to results directory")
    parser.add_argument("--api-key-path", type=str, default="./api.key",
                       help="path to openai api key file")
    # predefined model selection
    parser.add_argument("--vqa-model", type=str, default="gpt-4o",
                       help="vqa model variant")
    parser.add_argument("--grounding-model", type=str, default="groundingdino_swint_ogc", choices=["groundingdino_swint_ogc", "groundingdino_swinb_cogcoor"],
                       help="groundingdino model variant")
    parser.add_argument("--grounding-model-box-threshold", type=float, default=0.25,
                       help="groundingdino model box_threshold")
    parser.add_argument("--grounding-model-text-threshold", type=float, default=0.25,
                       help="groundingdino model text_threshold")
    parser.add_argument("--depth-model", type=str, default="unidepth-v2-vits14", choices=["unidepth-v2-vits14", "unidepth-v2-vitl14"],
                       help="unidepth model variant")
    # question loading and processing
    parser.add_argument("--num-questions", type=int, default=50,
                       help="number of questions to process per iteration (-1 for all)")
    parser.add_argument("--num-iterations", type=int, default=1,
                       help="number of iterations through the dataset")
    parser.add_argument("--programs-per-question", type=int, default=4,
                       help="number of program candidates per question")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="maximum parallel workers")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="random seed for reproducible sampling")
    parser.add_argument("--reorder-questions-per-iteration", action="store_true",
                       help="randomly reorder questions for each iteration")
    # example library and retrieval
    parser.add_argument("--quality-threshold", type=float, default=8.5,
                       help="minimum quality rating to add to example library")
    parser.add_argument("--similarity-threshold", type=float, default=0.8,
                       help="minimum similarity for example retrieval")
    parser.add_argument("--max-retrieved-examples", type=int, default=3,
                       help="maximum examples to retrieve")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-large-en-v1.5",
                       help="embedding model for question text similarity")
    # clustering and abstraction
    parser.add_argument("--abstraction-interval", type=int, default=10,
                       help="create tools every n questions")
    parser.add_argument("--clustering-threshold", type=float, default=0.8,
                       help="minimum similarity for initial embedding-based grouping")
    parser.add_argument("--min-cluster-size", type=int, default=4,
                       help="minimum cluster size for abstraction")
    parser.add_argument("--abstraction-potential-threshold", type=float, default=9.0,
                       help="minimum potential rating (0-10) to abstract")
    # abstraction validation
    parser.add_argument("--min-execution-success-rate", type=float, default=1.0,
                       help="minimum execution success rate")
    parser.add_argument("--min-correctness-rate", type=float, default=0.85,
                       help="minimum correctness rate over all examples in the cluster")
    parser.add_argument("--max-rewrite-retries", type=int, default=2,
                       help="maximum retries for program rewriting")
    parser.add_argument("--max-abstraction-retries", type=int, default=2,
                       help="maximum retries for tool abstraction")
    parser.add_argument("--max-merge-retries", type=int, default=2,
                       help="maximum retries for tool merging")
    # tool deduplication
    parser.add_argument("--enable-deduplication", type=bool, default=True,
                       help="enable tool deduplication")
    parser.add_argument("--deduplication-interval", type=int, default=10,
                       help="check for duplicate tools every n questions")
    parser.add_argument("--deduplication-threshold", type=float, default=0.95,
                       help="similarity threshold for merging tools")
    parser.add_argument("--min-tool-uses", type=int, default=0,
                       help="minimum uses before merge")
    parser.add_argument("--max-tool-uses", type=int, default=20,
                       help="maximum uses before merge")
    # example update
    parser.add_argument("--update-examples-after-abstraction", type=bool, default=True,
                       help="update example implementations in exmaple library after successful abstraction")
    parser.add_argument("--require-correctness-for-update", type=bool, default=True,
                       help="require successful correctness validation for example updates")
    parser.add_argument("--update-examples-after-deduplication", type=bool, default=True,
                       help="update example implementations in exmaple library after successful tool deduplication")
    # model selection
    parser.add_argument("--program-model", type=str, default="gpt-4o",
                       help="model for program generation")
    parser.add_argument("--judge-model", type=str, default="gpt-4o",
                       help="model for quality judging")
    parser.add_argument("--cluster-model", type=str, default="o4-mini",
                       help="model for cluster analysis -> reasoning models highly recommended")
    parser.add_argument("--abstraction-model", type=str, default="o4-mini",
                       help="model for tool abstraction")
    parser.add_argument("--deduplication-model", type=str, default="o4-mini",
                       help="model for tool deduplication")
    parser.add_argument("--correctness-model", type=str, default="gpt-4o",
                       help="model for correctness validation")
    parser.add_argument("--rewrite-model", type=str, default="o4-mini",
                       help="model for program rewriting")
    parser.add_argument("--merge-model", type=str, default="o4-mini",
                       help="model for tool merging")
    # temperatures
    parser.add_argument("--temperature-program", type=float, default=1.0,
                       help="temperature for program generation")
    parser.add_argument("--temperature-judge", type=float, default=0.0,
                       help="temperature for judging")
    parser.add_argument("--temperature-cluster", type=float, default=1.0,
                       help="temperature for cluster analysis")
    parser.add_argument("--temperature-abstraction", type=float, default=1.0,
                       help="temperature for abstraction")
    parser.add_argument("--temperature-deduplication", type=float, default=1.0,
                       help="temperature for deduplication")
    parser.add_argument("--temperature-correctness", type=float, default=0.0,
                       help="temperature for correctness validation")
    parser.add_argument("--temperature-rewrite", type=float, default=1.0,
                       help="temperature for program rewriting")
    parser.add_argument("--temperature-merge", type=float, default=1.0,
                       help="temperature for tool merging")
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                       help="reasoning.effort for reasoning models")
    # utils (resume, debugging, etc)
    parser.add_argument("--resume", type=str, default=None,
                       help="resume from a run id saved in results dir / start a new run id")
    parser.add_argument("--debug", action="store_true",
                       help="enable debug logging")
    parser.add_argument("--execution-timeout", type=int, default=300,
                       help="timeout (sec) for program execution")
    parser.add_argument("--save-exe-trace", action="store_true",
                       help="save full execution trace as html files for each program exeuction")
    args = parser.parse_args()
    # create config instance
    config = create_config_from_args(args)
    # setup logging
    setup_logging(config)
    log_config(config)
    # create and run pipeline
    pipeline = Pipeline(config)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted. State has been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
