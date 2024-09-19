import unittest
from unittest.mock import patch
import os
import sys
import metric_utils
import transformation_utils

from orchestrator.orchestrator import Orchestrator
# Get the directory of the current script
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
# Add the directory to sys.path
sys.path.append(script_dir)
from loguru import logger
from pipeline import Pipeline
import pandas as pd
import ast
import os
import shutil

class TestPipeline(unittest.TestCase):
    """Unit tests for the Pipeline class."""

    @staticmethod
    def delete_subfolders(parent_folder, subfolder_names):

        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(parent_folder, subfolder_name)
            if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
                shutil.rmtree(subfolder_path)
                print(f"Deleted subfolder: {subfolder_path}")
            else:
                print(f"Subfolder not found or not a directory: {subfolder_path}")


    def evaluate(self, additional_params, needs_breakdown, expected_tool_calls, expected_task_graph, actual_task_graph):
        pipeline = Pipeline(always_produce_task_graph=True, semantic_tool_filtering=True, include_indirect_dependencies=True)
        with patch.object(pipeline.orchestrator, 'breakdown_check', return_value=needs_breakdown):
            with patch.object(pipeline.executor, 'execute_query') as mock_execute_query:
                mock_execute_query.return_value = ("mock_response", None, expected_tool_calls)
                response = pipeline.execute_pipeline(additional_params["query"], actual_task_graph)
                if response == None:
                    logger.debug("Query " +str(additional_params["query"]) + "Couldn't be computed")
                    logger.debug("Deleting tools from folder" + str(expected_tool_calls))
                    parent_folder = "../../tools"
                    subfolder_names = ast.literal_eval(expected_tool_calls)[0]
                    self.delete_subfolders(parent_folder, subfolder_names)
                    return

                if needs_breakdown:
                    response_tool_calls = []
                    for task_info in response["tasks_info"]:
                        if task_info["tool_calls"] != ['no_tools_needed']:
                            for tool_call in task_info["tool_calls"]:
                                response_tool_calls.append(tool_call["func_name"])

                    metrics_df = metric_utils.save_task_graph_tools_and_response(additional_params,
                                                                                 expected_task_graph,
                                                                                 actual_task_graph,
                                                                                 expected_tool_calls,
                                                                                 response_tool_calls, response)

                    logger.debug("Metrics : " + metrics_df.to_string(index=False, justify='center'))

                else:
                    response_tool_calls = []
                    if response["tasks_info"]['tool_calls'] != ['no_tools_needed']:
                        for tool_call in response["tasks_info"]['tool_calls']:
                            response_tool_calls.append(tool_call["func_name"])

                    self.assertTrue(all(expected_item in response_tool_calls for expected_item in expected_tool_calls))

    def test_eval_asynchow_dataset_sequential(self):
        eval_df = pd.read_csv("../../../AsyncHow-Based-Agentic-Systems-Evaluation-Dataset/create_eval_dataset/eval_data_seq_df_50.csv")

        task_graphs = eval_df["Task Graph"].to_list()
        expected_tool_calls = eval_df["expected_tool_calls"].to_list()
        gold_standard_responses = eval_df["gold_standard_response"].to_list()
        gold_standard_responses = [ast.literal_eval(s) for s in gold_standard_responses]

        scenarios = eval_df["Scenario Name"].to_list()
        complexity_scores = eval_df["Complexity Score"].to_list()
        orchestrator = Orchestrator()

        idx_to_debug = 0
        for idx, (expected_task_graph, expected_tool_call) in enumerate(zip(task_graphs[idx_to_debug:], expected_tool_calls[idx_to_debug:])):

            idx = idx_to_debug
            logger.debug("CURRENT IDX" + str(idx))
            additional_params = {
                    "scenario_name":  scenarios[idx],
                    "complexity_score": complexity_scores[idx],
                    "query": scenarios[idx],
                    "gold_standard_response": gold_standard_responses[idx]
                }

            actual_task_graph = orchestrator.produce_task_graph(scenarios[idx])
            expected_task_graph = ast.literal_eval(expected_task_graph)

            self.evaluate(additional_params, True, expected_tool_call, expected_task_graph, actual_task_graph)
            idx_to_debug += 1


    def test_eval_asynchow_dataset_parallele(self):
        eval_df = pd.read_csv("../../../AsyncHow-Based-Agentic-Systems-Evaluation-Dataset/create_eval_dataset/eval_data_para_df_50.csv")

        task_graphs = eval_df["Task Graph"].to_list()
        expected_tool_calls = eval_df["expected_tool_calls"].to_list()
        gold_standard_responses = eval_df["gold_standard_response"].to_list()
        gold_standard_responses = [ast.literal_eval(s) for s in gold_standard_responses]

        scenarios = eval_df["Scenario Name"].to_list()
        complexity_scores = eval_df["Category"].to_list()
        orchestrator = Orchestrator()

        idx_to_debug = 0
        for idx, (expected_task_graph, expected_tool_call) in enumerate(zip(task_graphs[idx_to_debug:], expected_tool_calls[idx_to_debug:])):

            idx = idx_to_debug
            logger.debug("CURRENT IDX" + str(idx))
            additional_params = {
                    "scenario_name":  scenarios[idx],
                    "complexity_score": complexity_scores[idx],
                    "query": scenarios[idx],
                    "gold_standard_response": gold_standard_responses[idx]
                }

            actual_task_graph = orchestrator.produce_task_graph(scenarios[idx])
            expected_task_graph = ast.literal_eval(expected_task_graph)

            self.evaluate(additional_params, True, expected_tool_call, expected_task_graph, actual_task_graph)
            idx_to_debug += 1