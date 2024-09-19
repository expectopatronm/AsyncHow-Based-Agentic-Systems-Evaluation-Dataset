import pandas as pd


def create_parallel_graph(edges, tools):
    # Initialize the task graph structure
    task_graph = {
        "task_graph": {
            "nodes": [],
            "edges": []  # No edges are added in this case
        }
    }

    # Map tools to integer IDs
    for i, tool in enumerate(tools):
        task_id = i + 1  # IDs start from 1 and increment for each tool
        task_graph["task_graph"]["nodes"].append({"id": task_id, "label": tool})

    return task_graph


def create_seq_task_graph(edges, node_descriptions):
    """
    Creates a task graph from a list of edges and node descriptions, removing 'Start' and 'End' nodes.

    Args:
        edges (list of tuples): List of edges where each edge is a tuple (from_node, to_node).
        node_descriptions (list of str): List of task descriptions.

    Returns:
        dict: A dictionary in the task graph format.
    """
    # Map each node to a unique task ID
    task_map = {}
    node_list = []
    current_task_id = 1 # ASCII 'A'

    # Create nodes excluding "Start" and "End"
    for i, description in enumerate(node_descriptions):
        if description not in ["Start", "End"]:
            task_id = f"task_{current_task_id}"
            task_map[str(i+1)] = task_id
            node_list.append({'id': task_id, 'label': description})
            current_task_id += 1

    # Create edges
    edge_list = []
    for from_node, to_node in edges:
        if from_node in task_map and to_node in task_map:
            edge_list.append({'from': task_map[from_node], 'to': task_map[to_node]})

    assert len(edge_list) != 0

    # Return the task graph
    return {'task_graph': {'nodes': node_list, 'edges': edge_list}}


import random
def create_seq_graphs(df_name):

    df = pd.read_pickle("asynchow_data/seq_benchmark.pkl")

    # Specify the number of random samples you want to select
    num_samples = 100

    # Ensure that all lists have the same length
    list_length = len(next(iter(df.values())))

    # Generate random indices
    random_indices = random.sample(range(list_length), num_samples)

    # Initialize a new dictionary to store the random samples
    random_sampled_dict = {}

    # Iterate over each group (key) in the dictionary
    for key in ["titles", "steps", "edge_list"]:
        # Select elements from the list using the generated random indices
        random_samples = [df[key][i] for i in random_indices]
        # Store the samples in the new dictionary
        random_sampled_dict[key] = random_samples

    df_titles = random_sampled_dict["titles"]
    df_tools = random_sampled_dict["steps"]
    df_edges = random_sampled_dict['edge_list']
    task_graphs = []

    for title, tools, edges in zip(df_titles, df_tools, df_edges):
        task_graph = create_seq_task_graph(edges, tools)
        task_graphs.append(task_graph)

    metrics = {
        "Scenario Name": df_titles,
        "Tools": df_tools,
        "Task Graphs": task_graphs
    }

    new_metrics_df = pd.DataFrame(metrics)
    new_metrics_df.to_csv(df_name)


def create_parallel_graphs(df_name):

    df = pd.read_pickle("asynchow_data/para_benchmark.pkl")

    # Specify the number of random samples you want to select
    num_samples = 100

    # Ensure that all lists have the same length
    list_length = len(next(iter(df.values())))

    # Generate random indices
    random_indices = random.sample(range(list_length), num_samples)

    # Initialize a new dictionary to store the random samples
    random_sampled_dict = {}

    # Iterate over each group (key) in the dictionary
    for key in ["titles", "steps", "edge_list"]:
        # Select elements from the list using the generated random indices
        random_samples = [df[key][i] for i in random_indices]
        # Store the samples in the new dictionary
        random_sampled_dict[key] = random_samples

    df_titles = random_sampled_dict["titles"]
    df_tools = random_sampled_dict["steps"]
    df_edges = random_sampled_dict['edge_list']
    task_graphs = []

    for title, tools, edges in zip(df_titles, df_tools, df_edges):
        task_graph = create_parallel_graph(edges, tools)
        task_graphs.append(task_graph)

    metrics = {
        "Scenario Name": df_titles,
        "Tools": df_tools,
        "Task Graphs": task_graphs
    }

    new_metrics_df = pd.DataFrame(metrics)
    new_metrics_df.to_csv(df_name)


def create_async_graph(edges, node_descriptions):
    # Create a mapping from node names to integer IDs
    node_ids = {}
    current_id = 1

    # Map the nodes from the descriptions to integer IDs
    for i in range(len(node_descriptions)):
        node_ids[str(i + 1)] = current_id
        current_id += 1

    # Prepare the task graph format
    task_graph = {
        "task_graph": {
            "nodes": [],
            "edges": []
        }
    }

    # Add nodes with integer IDs and descriptions
    for i, desc in enumerate(node_descriptions):
        task_graph["task_graph"]["nodes"].append({"id": node_ids[str(i + 1)], "label": desc})

    # Add edges using integer IDs, excluding "START" and "END"
    for edge in edges:
        if edge[0] in node_ids and edge[1] in node_ids:
            task_graph["task_graph"]["edges"].append({
                "from": node_ids[edge[0]],
                "to": node_ids[edge[1]]
            })

    return task_graph


def create_async_graphs(df_name):

    df = pd.read_pickle("asynchow_data/async_benchmark.pkl")

    # Specify the number of random samples you want to select
    num_samples = 100

    # Ensure that all lists have the same length
    list_length = len(next(iter(df.values())))

    # Generate random indices
    random_indices = random.sample(range(list_length), num_samples)

    # Initialize a new dictionary to store the random samples
    random_sampled_dict = {}

    # Iterate over each group (key) in the dictionary
    for key in ["titles", "steps", "edge_list"]:
        # Select elements from the list using the generated random indices
        random_samples = [df[key][i] for i in random_indices]
        # Store the samples in the new dictionary
        random_sampled_dict[key] = random_samples

    df_titles = random_sampled_dict["titles"]
    df_tools = random_sampled_dict["steps"]
    df_edges = random_sampled_dict['edge_list']
    task_graphs = []

    for title, tools, edges in zip(df_titles, df_tools, df_edges):
        task_graph = create_async_graph(edges, tools)
        task_graphs.append(task_graph)

    metrics = {
        "Scenario Name": df_titles,
        "Tools": df_tools,
        "Task Graphs": task_graphs
    }

    new_metrics_df = pd.DataFrame(metrics)
    new_metrics_df.to_csv(df_name)

if __name__ == "__main__":
    #create_seq_graphs(df_name="asynchow_seq_df.csv")
    create_parallel_graphs(df_name="asynchow_para_df.csv")
    #create_async_graphs(df_name="asynchow_async_df.csv")
