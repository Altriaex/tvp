"""Offline reproduction tests; no model instances or service requests."""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import numpy as np
from openai import APIConnectionError, APITimeoutError, BadRequestError
from PIL import Image

from agents import (
    ClusterAnalyzer,
    CorrectnessJudge,
    ProgramGenerator,
    QualityJudge,
    ToolAbstractor,
    ToolDeduplicator,
)
from eval.eval_metrics import MetricEvaluator
from eval.extract_results import extract_results
from execution import ExecutionEngine, ExecutionValidator, ProgramRewriter
from library import ExampleLibrary, ToolLibrary
from models import (
    AbstractionInterval,
    Cluster,
    DeduplicationInterval,
    Example,
    ProgramCandidate,
    Question,
    Tool,
)
from pipeline import Pipeline
from predefined_apis import PredefinedModules
from state import PipelineState
from utils import LLMClient, ParallelExecutor


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        root = Path(".setup/tests")
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = SimpleNamespace(
            results_dir=self.root,
            resume_run_id="offline",
            run_id="offline",
            embedding_model="unused",
            num_questions=-1,
            num_iterations=3,
            stop_iteration=1,
            stop_position=2,
            random_seed=42,
            reorder_questions_per_iteration=True,
            abstraction_interval=1,
            deduplication_interval=1,
            enable_deduplication=True,
            annotations_path=str(self.root / "annotations.json"),
            max_workers=1,
            quality_threshold=8.5,
        )
        self.questions = [
            Question("image", i, "image.png", f"question {i}", "yes", "str")
            for i in range(4)
        ]
        Path(self.config.annotations_path).write_text(
            json.dumps([q.__dict__ for q in self.questions])
        )
        (self.root / "metadata.json").write_text(json.dumps({"num_iterations": 3}))
        self.answers = []

    def pipeline(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.config = self.config
        pipeline.state = PipelineState(self.config)
        pipeline.example_library = ExampleLibrary(self.config)
        pipeline.example_library.load(pipeline.state.examples)
        pipeline.tool_library = ToolLibrary(self.config)
        pipeline.tool_library.load(pipeline.state.tools)
        pipeline._global_cluster_counter = len(pipeline.state.clusters)

        def answer(question, iteration):
            self.answers.append((iteration, question.question_index))
            directory = (
                self.root
                / f"iterations/iteration_{iteration}/questions/q_{question.question_index}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "best_program.json").write_text(
                json.dumps({"selected": False, "reason": "offline failure"})
            )

        def abstract(iteration, interval):
            pipeline.state.abstraction_intervals.append(
                AbstractionInterval(
                    len(pipeline.state.abstraction_intervals) + 1, interval, interval
                )
            )

        def deduplicate(iteration, interval):
            pipeline.state.deduplication_intervals.append(
                DeduplicationInterval(
                    len(pipeline.state.deduplication_intervals) + 1, interval, interval
                )
            )

        pipeline._process_question = answer
        pipeline._abstract_and_deduplicate = abstract
        pipeline._perform_deduplication = deduplicate
        return pipeline

    def test_initial_selection_preserves_original_behavior(self):
        root = self.root
        for reorder in (False, True):
            with self.subTest(reorder=reorder):
                self.root = root / str(reorder)
                self.root.mkdir()
                self.config.results_dir = self.root
                self.config.num_questions = 2
                self.config.reorder_questions_per_iteration = reorder
                expected = self.questions
                if reorder:
                    expected = random.Random(self.config.random_seed + 1).sample(expected, len(expected))
                expected_ids = [q.question_index for q in expected[:2]]
                first = self.pipeline()._load_questions(1)
                self.assertEqual([q.question_index for q in first], expected_ids)
                saved = json.loads((self.root / "sampled_questions.json").read_text())
                self.assertEqual([q["question_index"] for q in saved], expected_ids)
                resumed = self.pipeline()._load_questions(1)
                self.assertEqual([q.question_index for q in resumed], expected_ids)
                second = self.pipeline()._load_questions(2)
                expected_second = random.Random(self.config.random_seed + 2).sample(expected_ids, 2) if reorder else expected_ids
                self.assertEqual([q.question_index for q in second], expected_second)

    def test_segments_orders_and_finished_restart(self):
        self.pipeline().run()
        saved = (self.root / "state/iteration_1_order.json").read_text()
        self.assertEqual(len(self.answers), 2)
        self.pipeline().run()
        self.assertEqual(len(self.answers), 2)
        self.config.stop_iteration, self.config.stop_position = 2, 1
        self.pipeline().run()
        self.assertEqual(len(self.answers), 5)
        self.assertEqual(
            saved, (self.root / "state/iteration_1_order.json").read_text()
        )
        self.config.stop_iteration, self.config.stop_position = 2, 4
        self.pipeline().run()
        self.assertEqual(len(self.answers), 8)
        self.assertFalse((self.root / "state/iteration_3_order.json").exists())
        self.config.stop_iteration = self.config.stop_position = None
        self.pipeline().run()
        self.assertEqual(len(self.answers), 12)
        self.pipeline().run()
        self.assertEqual(len(self.answers), 12)
        self.assertEqual(PipelineState(self.config).current_iteration, 4)

    def test_update_failure_reuses_answer_and_previous_stage(self):
        for failing_stage in ("abstraction", "deduplication"):
            with self.subTest(stage=failing_stage):
                pipeline = self.pipeline()
                method = (
                    "_abstract_and_deduplicate"
                    if failing_stage == "abstraction"
                    else "_perform_deduplication"
                )
                with (
                    patch.object(
                        pipeline,
                        method,
                        side_effect=RuntimeError("offline interruption"),
                    ),
                    self.assertRaises(RuntimeError),
                ):
                    pipeline.run()
                saved = PipelineState(self.config)
                self.assertEqual(saved.pending_stage, failing_stage)
                count = len(self.answers)
                self.config.stop_position = count
                self.pipeline().run()
                self.assertEqual(count, len(self.answers))
                self.assertEqual(PipelineState(self.config).pending_stage, "answer")
                self.config.stop_position = 2

    def test_published_state_commit_finishes_on_load(self):
        state = PipelineState(self.config)
        state.save()
        state.questions_processed = 1
        state.completed_questions = {"1": [0]}
        with (
            patch.object(
                state,
                "_finish_commit",
                side_effect=RuntimeError("interrupted promotion"),
            ),
            self.assertRaises(RuntimeError),
        ):
            state.save()
        restored = PipelineState(self.config)
        self.assertEqual(restored.questions_processed, 1)
        self.assertEqual(restored.completed_questions, {"1": [0]})
        self.assertFalse((restored.state_dir / ".commit").exists())

    def test_mutated_update_stage_restores_nonempty_state(self):
        root = self.root
        for stage in ("abstraction", "deduplication"):
            with self.subTest(stage=stage):
                self.root = root / stage
                self.root.mkdir()
                self.config.results_dir = self.root
                self.config.stop_position = 1
                state = PipelineState(self.config)
                state.examples = [
                    Example(
                        "q0",
                        self.questions[0],
                        "old program",
                        9,
                        {"final_result": 1},
                        np.array([1.0, 0.0]),
                    )
                ]
                state.tools = [
                    Tool("old_tool", 1, "def old_tool():", "old doc", "return 1")
                ]
                state.clusters = [
                    Cluster("cluster_0", ["q0"], "old pattern", "x", 9, "old reason")
                ]
                state.abstraction_intervals = [AbstractionInterval(1, 0, 0)]
                state.deduplication_intervals = [DeduplicationInterval(1, 0, 0)]
                state.abstraction_intervals_completed = state.deduplication_rounds = 1
                state.embedding_cache = {self.questions[0].question: [1.0, 0.0]}
                state.save()
                pipeline = self.pipeline()
                previous_files = {}

                def interrupt_after_mutation(
                    iteration,
                    interval,
                    pipeline=pipeline,
                    previous_files=previous_files,
                ):
                    # Capture the last committed stage, then leave partial updates in memory.
                    previous_files.update(
                        {
                            p.name: p.read_bytes()
                            for p in pipeline.state.state_dir.iterdir()
                            if p.is_file()
                        }
                    )
                    pipeline.example_library.examples[0].program = "unfinished program"
                    pipeline.example_library.examples[0].question_embedding[:] = [
                        0.0,
                        1.0,
                    ]
                    pipeline.tool_library.get_all_tools_including_deprecated()[
                        0
                    ].implementation = "return 99"
                    pipeline.state.clusters[0].pattern = "unfinished pattern"
                    pipeline.state.abstraction_intervals.append(
                        AbstractionInterval(99, 99, 99)
                    )
                    pipeline.state.deduplication_intervals.append(
                        DeduplicationInterval(99, 99, 99)
                    )
                    pipeline.state.abstraction_intervals_completed = 99
                    pipeline.state.deduplication_rounds = 99
                    pipeline.example_library._question_embedding_cache["unfinished"] = (
                        np.array([99.0])
                    )
                    raise RuntimeError("interrupted after state mutation")

                method = (
                    "_abstract_and_deduplicate"
                    if stage == "abstraction"
                    else "_perform_deduplication"
                )
                with (
                    patch.object(pipeline, method, interrupt_after_mutation),
                    self.assertRaises(RuntimeError),
                ):
                    pipeline.run()
                self.assertEqual(
                    pipeline.state.clusters[0].pattern, "unfinished pattern"
                )
                self.assertEqual(
                    pipeline.example_library.examples[0].program, "unfinished program"
                )
                answer_count = len(self.answers)
                resumed = self.pipeline()
                self.assertEqual(resumed.state.pending_stage, stage)
                self.assertEqual(resumed.state.completed_questions, {"1": []})
                self.assertEqual(resumed.state.questions_processed, 1)
                self.assertEqual(resumed.state.examples[0].program, "old program")
                self.assertEqual(resumed.state.tools[0].implementation, "return 1")
                self.assertEqual(resumed.state.clusters[0].pattern, "old pattern")
                np.testing.assert_array_equal(
                    resumed.state.examples[0].question_embedding, [1.0, 0.0]
                )
                self.assertEqual(resumed.state.abstraction_intervals_completed, 1)
                self.assertEqual(resumed.state.deduplication_rounds, 1)
                self.assertEqual(
                    len(resumed.state.abstraction_intervals),
                    1 if stage == "abstraction" else 2,
                )
                self.assertEqual(len(resumed.state.deduplication_intervals), 1)
                for name, content in previous_files.items():
                    self.assertEqual(
                        (resumed.state.state_dir / name).read_bytes(), content, name
                    )
                # Resume the pending stage without repeating the answer or completed updates.
                resumed.run()
                self.assertEqual(len(self.answers), answer_count)
                restored = PipelineState(self.config)
                self.assertEqual(restored.pending_stage, "answer")
                self.assertEqual(len(restored.completed_questions["1"]), 1)
                self.assertEqual(len(restored.abstraction_intervals), 2)
                self.assertEqual(len(restored.deduplication_intervals), 2)
                self.assertEqual(restored.examples[0].program, "old program")
                self.assertEqual(restored.tools[0].implementation, "return 1")
                self.assertEqual(restored.clusters[0].pattern, "old pattern")
                cache = json.loads(
                    (restored.state_dir / "embedding_cache.json").read_text()
                )
                self.assertNotIn("unfinished", cache["embeddings"])

    def test_partially_promoted_commit_recovers_all_new_files(self):
        root = self.root
        for promoted_count in (1, 4, 7):
            with self.subTest(promoted_files=promoted_count):
                self.config.results_dir = root / str(promoted_count)
                state = PipelineState(self.config)
                state.examples = [
                    Example(
                        "q0",
                        self.questions[0],
                        "old program",
                        9,
                        {"final_result": 1},
                        np.array([1.0, 0.0]),
                    )
                ]
                state.tools = [Tool("tool", 1, "def tool():", "doc", "return 1")]
                state.clusters = [
                    Cluster("cluster_0", ["q0"], "old pattern", "x", 9, "reason")
                ]
                state.abstraction_intervals = [AbstractionInterval(1, 0, 0)]
                state.deduplication_intervals = [DeduplicationInterval(1, 0, 0)]
                state.embedding_cache = {"question": [1.0, 0.0]}
                state.save()
                old_files = {p.name: p.read_bytes() for p in state.state_dir.iterdir()}
                state.examples[0].program = "new program"
                state.examples[0].execution_result = {"final_result": 2}
                state.examples[0].question_embedding = np.array([0.0, 1.0])
                state.tools[0].implementation = "return 2"
                state.clusters[0].pattern = "new pattern"
                state.abstraction_intervals.append(AbstractionInterval(2, 1, 1))
                state.deduplication_intervals.append(DeduplicationInterval(2, 1, 1))
                state.abstraction_intervals_completed = state.deduplication_rounds = 2
                state.embedding_cache = {"question": [0.0, 1.0]}
                state.questions_processed = 1
                state.completed_questions = {"1": [0]}
                expected_files = {}
                promoted = []
                replace = Path.replace

                def interrupt_during_promotion(
                    source,
                    target,
                    expected_files=expected_files,
                    replace=replace,
                    promoted=promoted,
                    promoted_count=promoted_count,
                ):
                    # Use the real replacement, then interrupt after only some files changed.
                    if not expected_files:
                        expected_files.update(
                            {p.name: p.read_bytes() for p in source.parent.iterdir()}
                        )
                    result = replace(source, target)
                    promoted.append(source.name)
                    if len(promoted) == promoted_count:
                        raise RuntimeError("interrupted partial promotion")
                    return result

                with (
                    patch.object(Path, "replace", interrupt_during_promotion),
                    self.assertRaises(RuntimeError),
                ):
                    state.save()
                remaining = set(expected_files) - set(promoted)
                self.assertTrue(remaining)
                self.assertEqual(len(promoted), promoted_count)
                for name in promoted:
                    self.assertEqual(
                        (state.state_dir / name).read_bytes(), expected_files[name]
                    )
                for name in remaining:
                    self.assertEqual(
                        (state.state_dir / name).read_bytes(), old_files[name]
                    )
                restored = PipelineState(self.config)
                self.assertFalse((state.state_dir / ".commit").exists())
                for name, content in expected_files.items():
                    self.assertEqual(
                        (state.state_dir / name).read_bytes(), content, name
                    )
                self.assertEqual(restored.questions_processed, 1)
                self.assertEqual(restored.completed_questions, {"1": [0]})
                self.assertEqual(restored.examples[0].program, "new program")
                self.assertEqual(
                    restored.examples[0].execution_result, {"final_result": 2}
                )
                np.testing.assert_array_equal(
                    restored.examples[0].question_embedding, [0.0, 1.0]
                )
                self.assertEqual(restored.tools[0].implementation, "return 2")
                self.assertEqual(restored.clusters[0].pattern, "new pattern")
                self.assertEqual(restored.abstraction_intervals_completed, 2)
                self.assertEqual(restored.deduplication_rounds, 2)
                self.assertEqual(len(restored.abstraction_intervals), 2)
                self.assertEqual(len(restored.deduplication_intervals), 2)

    def test_partial_evaluation_and_empty_categories(self):
        self.pipeline().run()
        extract_results("", self.root)
        data = json.loads((self.root / "eval/extracted_results.json").read_text())
        result = MetricEvaluator().evaluate_all_iterations(data)
        first = result["metrics_by_iteration"]["iter1"]
        self.assertEqual(first["total_questions"], 2)
        self.assertEqual(first["execution_failures"], 2)
        self.assertTrue(first["partial"])
        self.assertIsNone(first["accuracies"]["numeric_other"])
        self.assertEqual(result["metrics_by_iteration"]["iter2"]["total_questions"], 0)
        extract_results("", self.root)
        self.assertEqual(
            data, json.loads((self.root / "eval/extracted_results.json").read_text())
        )

    def test_evaluation_cli_outputs_stay_in_run(self):
        self.pipeline().run()
        for script in ("extract_results.py", "eval_metrics.py"):
            subprocess.run(
                [
                    sys.executable,
                    "eval/" + script,
                    "--results-path",
                    str(self.root.parent),
                    "--run-id",
                    self.root.name,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        summary = json.loads((self.root / "eval/metrics/summary.json").read_text())
        self.assertEqual(summary["summary"]["progress"]["iter1"]["completed"], 2)
        self.assertTrue((self.root / "eval/metrics/iter1_results.csv").exists())

    def test_unpublished_state_keeps_previous_stage(self):
        state = PipelineState(self.config)
        state.save()
        state.questions_processed = 1
        with (
            patch.object(
                state,
                "_save_tools",
                side_effect=RuntimeError("interrupted preparation"),
            ),
            self.assertRaises(RuntimeError),
        ):
            state.save()
        restored = PipelineState(self.config)
        self.assertEqual(restored.questions_processed, 0)
        restored.save()
        self.assertFalse((restored.state_dir / ".pending").exists())

    def test_response_and_network_exhaustion(self):
        client = LLMClient(self.config)
        client._client = Mock()
        client._client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=""), finish_reason="length"
                )
            ],
            usage=None,
        )
        self.assertEqual(
            client.generate("p", "qwen"),
            {"content": "", "finish_reason": "length", "usage": None},
        )
        request = httpx.Request("POST", "http://unused")
        for error in (
            APIConnectionError(request=request),
            APITimeoutError(request=request),
        ):
            client._client.chat.completions.create.side_effect = error
            with patch("utils.time.sleep"), self.assertRaises(APIConnectionError):
                client.generate("p", "qwen", max_retries=2)
            with self.assertRaises(APIConnectionError):
                ParallelExecutor(self.config).execute_parallel(
                    Mock(side_effect=error), [{}]
                )
        normal = Mock(side_effect=ValueError("ordinary failure"))
        self.assertEqual(
            ParallelExecutor(self.config).execute_parallel(normal, [{}]), [None]
        )

    def test_answer_interruption_discards_only_partial_question(self):
        pipeline = self.pipeline()
        original = pipeline._process_question
        question = pipeline._load_questions(1)[0]
        directory = (
            self.root / f"iterations/iteration_1/questions/q_{question.question_index}"
        )

        def interrupted(question, iteration):
            directory.mkdir(parents=True)
            (directory / "old_candidate.json").write_text("old")
            raise RuntimeError("interrupted answer")

        with (
            patch.object(pipeline, "_process_question", interrupted),
            self.assertRaises(RuntimeError),
        ):
            pipeline.run()
        self.config.stop_position = 1
        pipeline._process_question = original
        pipeline.run()
        self.assertFalse((directory / "old_candidate.json").exists())
        self.assertEqual(len(self.answers), 1)

    def test_example_update_failure_recovers_saved_example(self):
        pipeline = self.pipeline()
        question = pipeline._load_questions(1)[0]
        example = Example(
            "q0",
            question,
            "final_result = 1",
            9.0,
            {"final_result": 1},
            np.array([1.0, 0.0]),
        )
        answer = pipeline._process_question

        def with_example(question, iteration):
            answer(question, iteration)
            return example

        pipeline._process_question = with_example
        with (
            patch.object(
                pipeline.example_library,
                "add_example",
                side_effect=RuntimeError("update interrupted"),
            ),
            self.assertRaises(RuntimeError),
        ):
            pipeline.run()
        self.config.stop_position = 1
        resumed = self.pipeline()
        resumed.run()
        self.assertEqual(len(self.answers), 1)
        self.assertEqual(len(resumed.state.examples), 1)
        np.testing.assert_array_equal(
            PipelineState(self.config).examples[0].question_embedding, [1.0, 0.0]
        )

    def test_generation_vqa_and_reevaluation_network_boundaries(self):
        self.config.program_model = self.config.judge_model = self.config.vqa_model = (
            "qwen"
        )
        self.config.temperature_program = self.config.temperature_judge = 0.0
        self.config.image_path = str(self.root)
        image = Image.new("RGB", (2, 2))
        image.save(self.root / "image.png")
        error = APIConnectionError(request=httpx.Request("POST", "http://unused"))
        generator = ProgramGenerator(self.config)
        task = {
            "question": self.questions[0],
            "examples": [],
            "tools": [],
            "index": 0,
            "output_dir": self.root / "generation",
        }
        with (
            patch.object(generator.llm_client, "generate", side_effect=error),
            self.assertRaises(APIConnectionError),
        ):
            generator.generate_single(task)
        response = {"content": "", "finish_reason": "length", "usage": None}
        with patch.object(generator.llm_client, "generate", return_value=response):
            self.assertIsNone(generator.generate_single(task))
        self.assertEqual(
            json.loads((task["output_dir"] / "response_0.json").read_text()), response
        )
        http_error = BadRequestError(
            "bad request",
            response=httpx.Response(400, request=error.request),
            body=None,
        )
        with patch.object(generator.llm_client, "generate", side_effect=http_error):
            self.assertIsNone(generator.generate_single(task))
        modules = PredefinedModules.__new__(PredefinedModules)
        modules.config = self.config
        modules.llm_client = Mock()
        modules.execution_output_dir = self.root
        modules.vqa_call_index = 0
        modules.llm_client.generate_with_image.return_value = response
        self.assertEqual(modules._vqa_with_gpt(image, "question"), "")
        self.assertEqual(
            json.loads((self.root / "vqa_response_0.json").read_text()), response
        )
        null_response = {"content": None, "finish_reason": "length", "usage": None}
        modules.llm_client.generate_with_image.reset_mock()
        modules.llm_client.generate_with_image.return_value = null_response
        with patch("predefined_apis.time.sleep"):
            self.assertEqual(modules._vqa_with_gpt(image, "question"), "unknown")
        self.assertEqual(modules.llm_client.generate_with_image.call_count, 3)
        for index in range(1, 4):
            self.assertEqual(
                json.loads((self.root / f"vqa_response_{index}.json").read_text()),
                null_response,
            )
        modules.llm_client.generate_with_image.side_effect = error
        with self.assertRaises(APIConnectionError):
            modules._vqa_with_gpt(image, "question")
        library = ExampleLibrary(self.config)
        library.quality_judge = Mock()
        library.quality_judge.judge_single.side_effect = error
        example = Example(
            "q0",
            self.questions[0],
            "final_result=1",
            9,
            {"final_result": 1},
            np.array([1]),
        )
        with self.assertRaises(APIConnectionError):
            library._reevaluate_quality(example, 1)

    def test_remaining_roles_save_metadata_and_propagate_network(self):
        roles = [
            "judge",
            "cluster",
            "abstraction",
            "deduplication",
            "correctness",
            "rewrite",
            "merge",
        ]
        for role in roles:
            setattr(self.config, role + "_model", "qwen")
            setattr(self.config, "temperature_" + role, 0.0)
        self.config.min_cluster_size = 2
        self.config.deduplication_threshold = 0.95
        self.config.image_path = str(self.root)
        Image.new("RGB", (2, 2)).save(self.root / "image.png")
        example = Example(
            "q0",
            self.questions[0],
            "final_result = 1",
            9,
            {"final_result": 1},
            np.array([1]),
        )
        tool = Tool("tool", 1, "def tool():", "doc", "return 1")
        cluster = Cluster("cluster_0", ["q0"], "pattern", "parameters", 9, "reason")
        candidate = ProgramCandidate(self.questions[0], "final_result = 1", answer=1)
        calls = [
            (
                QualityJudge(self.config),
                "judge_single",
                [{"candidate": candidate, "output_dir": self.root / "judge"}],
                "generate_with_image",
                "judge/response.json",
            ),
            (
                ClusterAnalyzer(self.config),
                "analyze_group",
                [[example], self.root / "cluster"],
                "generate",
                "cluster/response.json",
            ),
            (
                ToolAbstractor(self.config),
                "abstract_from_cluster",
                [cluster, [example], [], [], self.root / "abstract"],
                "generate",
                "abstract/abstraction_response.json",
            ),
            (
                ToolDeduplicator(self.config),
                "find_duplicates",
                [[tool], self.root / "dedup"],
                "generate",
                "dedup/response.json",
            ),
            (
                ToolDeduplicator(self.config),
                "merge_tools",
                [[tool], "merge", "reason", self.root / "merge"],
                "generate",
                "merge/response.json",
            ),
            (
                CorrectnessJudge(self.config),
                "judge_correctness",
                [
                    {
                        "example_id": "q0",
                        "question": "q",
                        "original_answer": 1,
                        "new_answer": 2,
                        "original_program": "",
                        "rewritten_program": "",
                    },
                    self.root / "image.png",
                    self.root / "correct",
                ],
                "generate_with_image",
                "correct/response.json",
            ),
            (
                ProgramRewriter(self.config),
                "rewrite_with_tool",
                [example, tool, self.root / "rewrite"],
                "generate",
                "rewrite/rewrite_response_retry0.json",
            ),
        ]
        error = APITimeoutError(request=httpx.Request("POST", "http://unused"))
        response = {
            "content": None,
            "finish_reason": None,
            "usage": {"total_tokens": 3},
        }
        for agent, method, arguments, generate, record in calls:
            with self.subTest(role=method):
                with (
                    patch.object(agent.llm_client, generate, side_effect=error),
                    self.assertRaises(APIConnectionError),
                ):
                    getattr(agent, method)(*arguments)
                with patch.object(agent.llm_client, generate, return_value=response):
                    result = getattr(agent, method)(*arguments)
                if method == "judge_correctness":
                    self.assertEqual(result["verdict"], "ERROR")
                elif method in ("analyze_group", "find_duplicates"):
                    self.assertEqual(result, [])
                else:
                    self.assertIsNone(result)
                self.assertEqual(json.loads((self.root / record).read_text()), response)

    def test_cli_priority_metadata_and_temp_directory(self):
        import main
        from logging_utils import log_config

        captured = []

        def capture(args):
            captured.append(args)
            raise SystemExit(0)

        with (
            patch.dict(
                os.environ,
                {"TVP_PROGRAM_MODEL": "env-program", "TVP_JUDGE_MODEL": "env-judge"},
            ),
            patch.object(
                sys,
                "argv",
                ["main.py", "--resume", "offline", "--program-model", "cli-program"],
            ),
            patch.object(main, "create_config_from_args", capture),
            self.assertRaises(SystemExit),
        ):
            main.main()
        self.assertEqual(captured[0].program_model, "cli-program")
        self.assertEqual(captured[0].judge_model, "env-judge")
        first = (self.root / "metadata.json").read_text()
        log_config(self.config)
        self.assertEqual(first, (self.root / "metadata.json").read_text())

        def construct(config):
            self.assertEqual(os.environ["TMPDIR"], str((self.root / "tmp").resolve()))
            self.assertEqual(tempfile.tempdir, os.environ["TMPDIR"])
            return Mock()

        with (
            patch.dict(os.environ),
            patch.object(tempfile, "tempdir"),
            patch.object(sys, "argv", ["main.py", "--resume", "offline"]),
            patch.object(main, "create_config_from_args", return_value=self.config),
            patch.object(main, "setup_logging"),
            patch.object(main, "log_config"),
            patch("pipeline.Pipeline", side_effect=construct),
        ):
            main.main()

    def test_shared_engine_and_signal_cleanup(self):
        import signal

        engine = ExecutionEngine.__new__(ExecutionEngine)
        engine.timeout = 1
        path = self.root / "program.py"
        path.write_text("raise KeyboardInterrupt()")
        previous = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(KeyboardInterrupt):
            engine._run_program(path, {})
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.alarm(0), 0)
        path.write_text("raise failure")
        network = APIConnectionError(request=httpx.Request("POST", "http://unused"))
        with self.assertRaises(APIConnectionError):
            engine._run_program(path, {"failure": network})
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        error = engine._run_program(path, {"failure": ValueError("ordinary")})
        self.assertIn("ValueError", error["error"])
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        validator = ExecutionValidator(self.config, engine)
        self.assertIs(validator.execution_engine, engine)


if __name__ == "__main__":
    unittest.main()
