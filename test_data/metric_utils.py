import networkx as nx
from sentence_transformers import SentenceTransformer, util
from gpt.gpt_client import GPT4oAzureClientManager
import numpy as np
import pandas as pd
import os
import ast
import math
from tqdm import tqdm
from loguru import logger

model = SentenceTransformer('all-MiniLM-L6-v2')


def calculate_precision(tp, fp):
    """
    Calculates the Precision based on True Positives (TP) and False Positives (FP).

    Parameters:
        tp (int): Number of True Positives.
        fp (int): Number of False Positives.

    Returns:
        float: The Precision value.
    """
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def calculate_recall(tp, fn):
    """
    Calculates the Recall based on True Positives (TP) and False Negatives (FN).

    Parameters:
        tp (int): Number of True Positives.
        fn (int): Number of False Negatives.

    Returns:
        float: The Recall value.
    """
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def calculate_f1(precision, recall):
    """
    Calculates the F1 Score based on Precision and Recall.

    Parameters:
        precision (float): The Precision value.
        recall (float): The Recall value.

    Returns:
        float: The F1 Score.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_similarity_with_llm(gold_standard_answer, actual_answer):
    prompt = f"""You are an expert in evaluating text similarity. Please compare the following two answers and provide a similarity score from 0 to 100, where 100 means the answers are identical in meaning and 0 means they are completely different.

    Gold Standard Answer: "{gold_standard_answer}"

    Actual Answer: "{actual_answer}"

    Provide only the similarity score as a number:
    """

    evaluator = GPT4oAzureClientManager()
    result = evaluator.query_gpt_4o(prompt)
    return float(result)


def calculate_label_similarity(expected_labels, actual_labels):
    expected_embeddings = model.encode(expected_labels, convert_to_tensor=True)
    actual_embeddings = model.encode(actual_labels, convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(expected_embeddings, actual_embeddings).numpy()
    return similarity_matrix


def match_nodes(expected_graph, actual_graph):
    expected_labels = [node['label'] for node in expected_graph['nodes']]
    actual_labels = [node['label'] for node in actual_graph['nodes']]

    similarity_matrix = calculate_label_similarity(expected_labels, actual_labels)

    matched_pairs = []
    matched_ids = set()

    for i, expected_node in enumerate(expected_graph['nodes']):
        best_match_index = np.argmax(similarity_matrix[i])
        best_match_similarity = similarity_matrix[i][best_match_index]

        if best_match_index not in matched_ids:
            matched_ids.add(best_match_index)
            matched_pairs.append(
                (expected_node['id'], actual_graph['nodes'][best_match_index]['id'], best_match_similarity))

    tp = len(matched_pairs)
    fp = len(actual_graph['nodes']) - tp
    fn = len(expected_graph['nodes']) - tp

    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1 = calculate_f1(precision, recall)

    return matched_pairs, similarity_matrix, precision, recall, f1



def match_edges(expected_graph, actual_graph, matched_pairs):
    expected_edges = {(edge['from'], edge['to']) for edge in expected_graph['edges']}
    actual_edges = {(edge['from'], edge['to']) for edge in actual_graph['edges']}

    matched_edges = 0
    for expected_edge in expected_edges:
        expected_from, expected_to = expected_edge
        matched_from = None
        matched_to = None

        for match in matched_pairs:
            if match[0] == expected_from:
                matched_from = match[1]
            if match[0] == expected_to:
                matched_to = match[1]

        if (matched_from, matched_to) in actual_edges:
            matched_edges += 1

    tp = matched_edges
    fp = len(actual_edges) - matched_edges
    fn = len(expected_edges) - matched_edges

    # Calculate precision, recall, and F1 score using the provided functions
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1 = calculate_f1(precision, recall)

    return matched_edges, precision, recall, f1


def calculate_graph_edit_distance(expected_graph, actual_graph):
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()

    for node in expected_graph['nodes']:
        G1.add_node(node['id'])
    for edge in expected_graph['edges']:
        G1.add_edge(edge['from'], edge['to'])

    for node in actual_graph['nodes']:
        G2.add_node(node['id'])
    for edge in actual_graph['edges']:
        G2.add_edge(edge['from'], edge['to'])

    edit_distance = nx.graph_edit_distance(G1, G2)
    return edit_distance


def structural_similarity_index(node_similarity_matrix, edge_f1):
    node_similarity_score = np.mean(np.max(node_similarity_matrix, axis=1))
    structural_similarity = (node_similarity_score + edge_f1) / 2
    return structural_similarity


def calculate_path_length_similarity(expected_tg, actual_tg):
    def get_paths(graph):
        paths = {}
        nodes = {node['id'] for node in graph['nodes']}
        for node in nodes:
            paths[node] = {}
            for target_node in nodes:
                if node == target_node:
                    paths[node][target_node] = 0
                else:
                    paths[node][target_node] = float('inf')

        for edge in graph['edges']:
            paths[edge['from']][edge['to']] = 1

        for k in nodes:
            for i in nodes:
                for j in nodes:
                    if paths[i][j] > paths[i][k] + paths[k][j]:
                        paths[i][j] = paths[i][k] + paths[k][j]
        return paths

    # Ensure the inputs are dictionaries, not strings
    if isinstance(expected_tg, str):
        expected_tg = ast.literal_eval(expected_tg)
    if isinstance(actual_tg, str):
        actual_tg = ast.literal_eval(actual_tg)

    paths_g1 = get_paths(expected_tg)
    paths_g2 = get_paths(actual_tg)

    # Ensure both paths dictionaries have the same nodes
    all_nodes = set(paths_g1.keys()).union(paths_g2.keys())

    for node in all_nodes:
        if node not in paths_g1:
            paths_g1[node] = {n: float('inf') for n in all_nodes}
            paths_g1[node][node] = 0
        if node not in paths_g2:
            paths_g2[node] = {n: float('inf') for n in all_nodes}
            paths_g2[node][node] = 0

        # Also ensure that each node dictionary includes all other nodes
        for target_node in all_nodes:
            if target_node not in paths_g1[node]:
                paths_g1[node][target_node] = float('inf')
            if target_node not in paths_g2[node]:
                paths_g2[node][target_node] = float('inf')

    total_paths = 0
    similar_paths = 0

    for source in paths_g1:
        for target in paths_g1[source]:
            if paths_g1[source][target] == paths_g2[source].get(target, float('inf')):
                similar_paths += 1
            total_paths += 1

    similarity = similar_paths / total_paths if total_paths > 0 else 0
    return similarity


def calculate_node_coverage(expected_task_graph, actual_task_graph):

    # Ensure the inputs are dictionaries, not strings
    if isinstance(expected_task_graph, str):
        expected_task_graph = ast.literal_eval(expected_task_graph)
    if isinstance(actual_task_graph, str):
        actual_task_graph = ast.literal_eval(actual_task_graph)

    # Extract node IDs
    expected_nodes = {node['id'] for node in expected_task_graph['nodes']}
    actual_nodes = {node['id'] for node in actual_task_graph['nodes']}

    # Calculate matched nodes
    matched_nodes = expected_nodes & actual_nodes

    # Calculate coverage
    coverage = len(matched_nodes) / len(expected_nodes) if len(expected_nodes) > 0 else 0

    return coverage


def calculate_edge_coverage(expected_task_graph, actual_task_graph):
    """
    Calculates the edge coverage between two task graphs.

    Args:
        expected_task_graph (dict or str): The expected task graph, either as a dictionary or a string.
        actual_task_graph (dict or str): The actual task graph, either as a dictionary or a string.

    Returns:
        float: The coverage of edges in the actual task graph relative to the expected task graph.
    """
    # Ensure the inputs are dictionaries, not strings
    if isinstance(expected_task_graph, str):
        expected_task_graph = ast.literal_eval(expected_task_graph)
    if isinstance(actual_task_graph, str):
        actual_task_graph = ast.literal_eval(actual_task_graph)

    # Extract edge sets
    expected_edges = {(edge['from'], edge['to']) for edge in expected_task_graph['edges']}
    actual_edges = {(edge['from'], edge['to']) for edge in actual_task_graph['edges']}

    # Calculate matched edges
    matched_edges = expected_edges & actual_edges

    # Calculate coverage
    coverage = len(matched_edges) / len(expected_edges) if len(expected_edges) > 0 else 0

    return coverage


def calculate_weighted_f1_score(matched_pairs, total_expected, total_actual, weights):
    weighted_sum = sum(weights[node] for node in matched_pairs)
    precision = weighted_sum / sum(weights[node] for node in total_actual) if total_actual else 0
    recall = weighted_sum / sum(weights[node] for node in total_expected) if total_expected else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def calculate_task_difficulty_score(task_difficulty_dict, actual_task_graph):
    total_difficulty = sum(task_difficulty_dict.get(node['id'], 1.0) for node in actual_task_graph['nodes'])
    average_difficulty = total_difficulty / len(actual_task_graph['nodes'])
    return average_difficulty



def calculate_error_propagation(task_graph, error_nodes):
    def find_descendants(node_id, edges):
        descendants = set()
        for edge in edges:
            if edge['from'] == node_id:
                descendants.add(edge['to'])
                descendants.update(find_descendants(edge['to'], edges))
        return descendants

    error_influence = {}

    for node in error_nodes:
        descendants = find_descendants(node, task_graph['edges'])
        for desc in descendants:
            error_influence[desc] = error_influence.get(desc, 0) + 1

    total_influence = sum(error_influence.values())
    normalized_influence = {node: count / total_influence for node, count in error_influence.items()}

    return normalized_influence


def calculate_centrality_measures(task_graph):
    centrality = {}
    node_counts = {node['id']: 0 for node in task_graph['nodes']}

    for edge in task_graph['edges']:
        node_counts[edge['from']] += 1
        node_counts[edge['to']] += 1

    total_count = sum(node_counts.values())
    centrality['betweenness'] = {node: count / total_count for node, count in node_counts.items()}

    closeness_centrality = {}
    for node in task_graph['nodes']:
        closeness_sum = 0
        for other_node in task_graph['nodes']:
            if node['id'] != other_node['id']:
                closeness_sum += nx.shortest_path_length(task_graph, source=node['id'], target=other_node['id'])
        closeness_centrality[node['id']] = (len(task_graph['nodes']) - 1) / closeness_sum

    centrality['closeness'] = closeness_centrality

    return centrality


def calculate_subgraph_centrality(task_graph):
    subgraph_centrality = {node['id']: 0 for node in task_graph['nodes']}
    for node in task_graph['nodes']:
        # Use a simplified version of subgraph centrality focusing on direct neighbors
        neighbors = set(edge['to'] for edge in task_graph['edges'] if edge['from'] == node['id'])
        subgraph_centrality[node['id']] = len(neighbors)

    return subgraph_centrality


def calculate_graph_entropy(task_graph):
    degrees = {}
    for node in task_graph['nodes']:
        degrees[node['id']] = 0

    for edge in task_graph['edges']:
        degrees[edge['from']] += 1
        degrees[edge['to']] += 1

    total_degree = sum(degrees.values())
    probabilities = [d / total_degree for d in degrees.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy



def transform_task_graph_no_start_end(task_graph):
    # Create a new task graph dictionary with nodes and empty edges
    transformed_graph = {
        "task_graph": {
            "nodes": [],
            "edges": []  # Ensure that the edges list is empty
        }
    }

    # Transform nodes by mapping them to new IDs starting from 0, excluding 'START' and 'END'
    node_mapping = {}
    new_id = 0
    for node in task_graph["task_graph"]["nodes"]:
        if node["label"] not in ["START", "END"]:
            node_mapping[node["id"]] = new_id
            transformed_graph["task_graph"]["nodes"].append({
                "id": new_id,
                "label": node["label"]
            })
            new_id += 1

    # No need to transform edges since the edges list should be empty

    return transformed_graph


def save_task_graph_tools_and_response(additional_params, expected_graph, actual_graph, expected_tools, actual_tools, response):
    expected_tools = ast.literal_eval(expected_tools)[0]
    actual_tools = set(actual_tools)

    metrics = {
        "Scenario_Name": additional_params["scenario_name"],
        "Complexity Score": additional_params["complexity_score"],
        "Query": additional_params["query"],
        "Expected TG": expected_graph,
        "Actual TG": actual_graph,
        "Expected Tools": expected_tools,
        "Actual Tools": actual_tools,
        "Expected Response": additional_params["gold_standard_response"][0],
        "Actual Response": response['final_response'],
        "Model response": response
    }

    new_metrics_df = pd.DataFrame([metrics])
    #df_path_pkl = "df_seq_task_graphs_and_tools_50.pkl"
    #df_path_csv = "df_seq_task_graphs_and_tools_50.csv"

    df_path_pkl = "df_para_task_graphs_and_tools_50.pkl"
    df_path_csv = "df_para_task_graphs_and_tools_50.csv"


    if os.path.exists(df_path_pkl):
        metrics_df = pd.read_pickle(df_path_pkl)
        metrics_df = pd.concat([metrics_df, new_metrics_df], ignore_index=True)
    else:
        metrics_df = new_metrics_df

    metrics_df.to_pickle(df_path_pkl)
    metrics_df.to_csv(df_path_csv)
    return new_metrics_df


def create_metrics_seq():

    df = pd.read_csv("df_seq_task_graphs_and_tools_50.csv")
    df = df.drop(columns=['Unnamed: 0'])
    scenario_names = df["Scenario_Name"].to_list()
    complexity_scores = df["Complexity Score"].to_list()
    queries= df["Query"].to_list()

    expected_task_graphs = df["Expected TG"].to_list()
    actual_task_graphs = df["Actual TG"].to_list()
    expected_tools = df["Expected Tools"].to_list()
    actual_tools = df["Actual Tools"].to_list()
    expected_responses = df["Expected Response"].to_list()
    actual_responses = df["Actual Response"].to_list()
    model_responses = df["Model response"].to_list()

    for scenario_name, complex_score, query, expected_graph, actual_graph, expected_tool, actual_tool, exp_resp, actual_resp, model_resp \
            in tqdm(zip(scenario_names, complexity_scores, queries, expected_task_graphs,actual_task_graphs, expected_tools, actual_tools, expected_responses, actual_responses, model_responses )):



        expected_graph = ast.literal_eval(expected_graph)
        expected_graph= expected_graph['task_graph']
        actual_graph = ast.literal_eval(actual_graph)
        expected_tool = ast.literal_eval(expected_tool)

        actual_tool = ast.literal_eval(actual_tool)
        actual_tool = list(actual_tool)

        matched_pairs, node_similarity_matrix, node_precision, node_recall, node_f1 = match_nodes(expected_graph,
                                                                                                  actual_graph)

        matched_edges, edge_precision, edge_recall, edge_f1 = match_edges(expected_graph, actual_graph, matched_pairs)
        graph_edit_distance = calculate_graph_edit_distance(expected_graph, actual_graph)
        structural_similarity = structural_similarity_index(node_similarity_matrix, edge_f1)
        node_label_similarity = np.mean(np.max(node_similarity_matrix, axis=1))

        tp = len(set(expected_tool) & set(actual_tool))
        fp = len(set(actual_tool) - set(expected_tool))
        fn = len(set(expected_tool) - set(actual_tool))

        tool_precision = calculate_precision(tp, fp)
        tool_recall = calculate_recall(tp, fn)
        tool_f1 = calculate_f1(tool_precision, tool_recall)

        answer_score = evaluate_similarity_with_llm(exp_resp,actual_resp)

        path_length_similarity = calculate_path_length_similarity(expected_graph, actual_graph)
        node_coverage = calculate_node_coverage(expected_graph, actual_graph)
        edge_coverage = calculate_edge_coverage(expected_graph, actual_graph)

        execution_time = 0
        model_resp = ast.literal_eval(model_resp)
        for task in model_resp["tasks_info"]:
            execution_time = task["execution_timing"]["duration"]

        expected_task_graph_complexity = compute_task_graph_complexity(expected_graph)
        actual_task_graph_complexity = compute_task_graph_complexity(actual_graph)

        metrics = {
            "Scenario Name": scenario_name,
            "Complexity Score": complex_score,
            "Query": query,
            "Expected TG": expected_graph,
            "Actual TG": actual_graph,
            "Expected Task Complexity": expected_task_graph_complexity,
            "Actual Task Complexity": actual_task_graph_complexity,
            "Expected Tools": expected_tools,
            "Actual Tools": actual_tools,
            "Node Label Similarity": node_label_similarity,
            "Node Precision": node_precision,
            "Node Recall": node_recall,
            "Node F1 Score": node_f1,
            "Matched Pairs": len(matched_pairs),
            "Matched Edges": matched_edges,
            "Edge Precision": edge_precision,
            "Edge Recall": edge_recall,
            "Edge F1 Score": edge_f1,
            "Graph Edit Distance": graph_edit_distance,
            "Structural Similarity Index": structural_similarity,
            "Path Length Similarity": path_length_similarity,
            "Node coverage": node_coverage,
            "Edge coverage": edge_coverage,
            "Tool Precision": tool_precision,
            "Tool Recall": tool_recall,
            "Tool F1 Score": tool_f1,
            "Answer Score": answer_score,
            "Gold Standard Answer": exp_resp,
            "Execution Time": execution_time
        }

        new_metrics_df = pd.DataFrame([metrics])
        df_path_pkl = "metrics_df_seq_final_50.pkl"
        df_path_csv = "metrics_df_seq_final_50.csv"

        # Check if the DataFrame already exists
        if os.path.exists(df_path_pkl):
            # If it exists, load it and append the new row
            metrics_df = pd.read_pickle(df_path_pkl)
            metrics_df = pd.concat([metrics_df, new_metrics_df], ignore_index=True)

        else:
            # If it doesn't exist, use the new DataFrame
            metrics_df = new_metrics_df

        # Save the updated DataFrame back to the file
        metrics_df.to_pickle(df_path_pkl)
        metrics_df.to_csv(df_path_csv)


def create_metrics_para():

    df = pd.read_csv("df_para_task_graphs_and_tools_50.csv")
    df = df.drop(columns=['Unnamed: 0'])
    scenario_names = df["Scenario_Name"].to_list()
    complexity_scores = df["Complexity Score"].to_list()
    queries= df["Query"].to_list()

    #Only for parallele graphs !!!!
    expected_task_graphs = df["Expected TG"].to_list()
    expected_task_graphs = [transform_task_graph_no_start_end(ast.literal_eval(task_graph)) for task_graph in expected_task_graphs]

    actual_task_graphs = df["Actual TG"].to_list()
    expected_tools = df["Expected Tools"].to_list()
    actual_tools = df["Actual Tools"].to_list()
    expected_responses = df["Expected Response"].to_list()
    actual_responses = df["Actual Response"].to_list()
    model_responses = df["Model response"].to_list()

    for scenario_name, complex_score, query, expected_graph, actual_graph, expected_tool, actual_tool, exp_resp, actual_resp, model_resp \
            in tqdm(zip(scenario_names, complexity_scores, queries, expected_task_graphs,actual_task_graphs, expected_tools, actual_tools, expected_responses, actual_responses, model_responses )):

        #expected_graph = ast.literal_eval(expected_graph)
        expected_graph= expected_graph['task_graph']
        actual_graph = ast.literal_eval(actual_graph)
        expected_tool = ast.literal_eval(expected_tool)

        actual_tool = ast.literal_eval(actual_tool)
        actual_tool = list(actual_tool)


        matched_pairs, node_similarity_matrix, node_precision, node_recall, node_f1 = match_nodes(expected_graph,
                                                                                                  actual_graph)

        matched_edges, edge_precision, edge_recall, edge_f1 = match_edges(expected_graph, actual_graph, matched_pairs)
        graph_edit_distance = calculate_graph_edit_distance(expected_graph, actual_graph)
        structural_similarity = structural_similarity_index(node_similarity_matrix, edge_f1)
        node_label_similarity = np.mean(np.max(node_similarity_matrix, axis=1))

        tp = len(set(expected_tool) & set(actual_tool))
        fp = len(set(actual_tool) - set(expected_tool))
        fn = len(set(expected_tool) - set(actual_tool))

        tool_precision = calculate_precision(tp, fp)
        tool_recall = calculate_recall(tp, fn)
        tool_f1 = calculate_f1(tool_precision, tool_recall)

        answer_score = evaluate_similarity_with_llm(exp_resp,actual_resp)

        path_length_similarity = calculate_path_length_similarity(expected_graph, actual_graph)
        node_coverage = calculate_node_coverage(expected_graph, actual_graph)
        edge_coverage = calculate_edge_coverage(expected_graph, actual_graph)

        execution_time = 0
        model_resp = ast.literal_eval(model_resp)
        for task in model_resp["tasks_info"]:
            execution_time = task["execution_timing"]["duration"]

        expected_task_graph_complexity = compute_task_graph_complexity(expected_graph)
        actual_task_graph_complexity = compute_task_graph_complexity(actual_graph)


        metrics = {
            "Scenario Name": scenario_name,
            "Complexity Score": complex_score,
            "Query": query,
            "Expected TG": expected_graph,
            "Actual TG": actual_graph,
            "Expected Task Complexity": expected_task_graph_complexity,
            "Actual Task Complexity": actual_task_graph_complexity,
            "Expected Tools": expected_tools,
            "Actual Tools": actual_tools,
            "Node Label Similarity": node_label_similarity,
            "Node Precision": node_precision,
            "Node Recall": node_recall,
            "Node F1 Score": node_f1,
            "Matched Pairs": len(matched_pairs),
            "Matched Edges": matched_edges,
            "Edge Precision": edge_precision,
            "Edge Recall": edge_recall,
            "Edge F1 Score": edge_f1,
            "Graph Edit Distance": graph_edit_distance,
            "Structural Similarity Index": structural_similarity,
            "Path Length Similarity": path_length_similarity,
            "Node coverage": node_coverage,
            "Edge coverage": edge_coverage,
            "Tool Precision": tool_precision,
            "Tool Recall": tool_recall,
            "Tool F1 Score": tool_f1,
            "Answer Score": answer_score,
            "Gold Standard Answer": exp_resp,
            "Execution Time": execution_time
        }

        new_metrics_df = pd.DataFrame([metrics])

        df_path_pkl = "metrics_df_para_final_50.pkl"
        df_path_csv = "metrics_df_para_final_50.csv"

        # Check if the DataFrame already exists
        if os.path.exists(df_path_pkl):
            # If it exists, load it and append the new row
            metrics_df = pd.read_pickle(df_path_pkl)
            metrics_df = pd.concat([metrics_df, new_metrics_df], ignore_index=True)

        else:
            # If it doesn't exist, use the new DataFrame
            metrics_df = new_metrics_df

        # Save the updated DataFrame back to the file
        metrics_df.to_pickle(df_path_pkl)
        metrics_df.to_csv(df_path_csv)


def compute_task_graph_complexity(task_graph):
    # Number of nodes
    num_nodes = len(task_graph['nodes'])

    # Number of edges
    num_edges = len(task_graph['edges'])

    # Complexity is the sum of the number of nodes and edges
    complexity = num_nodes + num_edges

    return int(complexity)


if __name__ == "__main__":
    #create_metrics_para()
    create_metrics_seq()

    #df = pd.read_csv("metrics_df_para_final_50.csv")
    df = pd.read_csv("metrics_df_seq_final_50.csv")

    df = df.drop(columns=['Query', 'Agent response', 'Gold Standard Answer', 'Expected TG', 'Actual TG','Expected Tools','Actual Tools', "Unnamed: 0.2","Unnamed: 0.1","Unnamed: 0", "Unnamed: 0.3"], errors='ignore')
    complexity_scores = df["Complexity Score"].to_list()
    scores = [ast.literal_eval(score)[0] for score in complexity_scores]
    df["Complexity Score"] = scores

    df.rename(columns={'Complexity Score': 'Category'}, inplace=True)

    category_mapping = {
        "Food and Entertainment": 1,
        "Education and Communication": 2,
        "Family Life": 3,
        "Home and Garden": 4,
        "Health": 5,
        "Others": 6
    }

    df["Category"] = df["Category"].map(category_mapping)
    #df.to_csv("refined_metrics_df_para_final_50.csv")
    df.to_csv("refined_metrics_df_seq_final_50.csv")

