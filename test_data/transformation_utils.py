import re
import pandas as pd

def transform_to_structured_tg(task_graph):
    nodes = []
    edges = []
    task_id_map = {}

    # Assign unique IDs to each task
    for index, task in enumerate(task_graph):
        task_id = f"task_{index + 1}"
        task_id_map[task['Task Name']] = task_id
        nodes.append({
            "id": task_id,
            "label": task['Task Description']
        })

    # Create edges based on dependencies
    for task in task_graph:
        current_task_id = task_id_map[task['Task Name']]
        for dependency in task['Dependencies']:
            dependency_id = task_id_map.get(dependency.strip())
            if dependency_id:
                edges.append({
                    "from": dependency_id,
                    "to": current_task_id
                })

    return {
        "task_graph": {
            "nodes": nodes,
            "edges": edges
        }
    }

def transform_to_df(input_string):
    # Extracting each scenario
    scenarios = re.split(r"### Scenario \d+: ", input_string)[1:]

    # Prepare list to hold scenario dictionaries
    scenario_data = []

    for scenario in scenarios:
        scenario_dict = {}

        # Extract Scenario Name
        scenario_name_match = re.search(r"\*\*Scenario Name:\*\* (.*?)\n", scenario)
        if scenario_name_match:
            scenario_dict["Scenario Name"] = scenario_name_match.group(1).strip()

        # Extract Complexity Score
        complexity_score_match = re.search(r"\*\*Complexity Score:\*\* (\d+)", scenario)
        if complexity_score_match:
            scenario_dict["Complexity Score"] = int(complexity_score_match.group(1).strip())

        # Extract User Query
        user_query_match = re.search(r"\*\*User Query:\*\* \"(.*?)\"\n", scenario)
        if user_query_match:
            scenario_dict["User Query"] = user_query_match.group(1).strip()

        # Extract Task Graph
        task_graph_match = re.findall(
            r"- \*\*Task Name:\*\* (.*?)\n\s+- \*\*Task Description:\*\* (.*?)\n\s+- \*\*Dependencies:\*\* \[(.*?)\]",
            scenario,
        )
        task_graph = []
        for task in task_graph_match:
            dependencies = task[2].strip()
            dependencies_list = (
                [dep.strip() for dep in dependencies.split(",")] if dependencies else []
            )
            task_graph.append({
                "Task Name": task[0].strip(),
                "Task Description": task[1].strip(),
                "Dependencies": dependencies_list,
            })
        scenario_dict["Task Graph"] = task_graph

        # Extract Tool List
        tool_list_match = re.findall(
            r"- \*\*Tool Name:\*\* (.*?)\n\s+- \*\*Tool Description:\*\* (.*?)\n", scenario
        )
        tool_list = [
            {"Tool Name": tool[0].strip(), "Tool Description": tool[1].strip()}
            for tool in tool_list_match
        ]
        scenario_dict["Tool List"] = tool_list

        # Extract Tool Selection
        tool_selection_match = re.findall(
            r"- \*\*Task Name:\*\* (.*?)\n\s+- \*\*Selected Tool:\*\* (.*?)\n\s+- \*\*Reason for Selection:\*\* (.*?)\n",
            scenario,
        )
        tool_selection = [
            {
                "Task Name": tool_sel[0].strip(),
                "Selected Tool": tool_sel[1].strip(),
                "Reason for Selection": tool_sel[2].strip(),
            }
            for tool_sel in tool_selection_match
        ]
        scenario_dict["Tool Selection"] = tool_selection

        scenario_data.append(scenario_dict)

    # Convert to DataFrame
    df = pd.DataFrame(scenario_data)
    return df