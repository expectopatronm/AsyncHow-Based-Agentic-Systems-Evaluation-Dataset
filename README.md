
# AsyncHow Agentic Systems Evaluation Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13799791.svg)](https://doi.org/10.5281/zenodo.13799791)

## Overview

This repository hosts the **AsyncHow Agentic Systems Evaluation Dataset**, a comprehensive dataset created to evaluate the performance of agentic systems driven by Large Language Models (LLMs). 
The dataset is based on the work of Lin et al. [Graph-enhanced Large Language Models in Asynchronous Plan Reasoning] (https://github.com/fangru-lin/graph-llm-asynchow-plan).
This dataset is designed to assess dynamic task decomposition, tool selection, and task execution across various domains, enabling researchers to analyze agentic systems' behavior with both simple and complex tasks.
The dataset is foundational to our NeurIPS 2024 paper titled _Advancing Agentic Systems: Dynamic Task Decomposition, Tool Integration, and Evaluation using Novel Metrics and Dataset_. It was developed as part of our framework for evaluating agentic systems, which leverages novel metrics such as Node F1 Score, Structural Similarity Index (SSI), and Tool F1 Score.

Additionally, the dataset includes **synthetic Python functions** generated using LLMs, which mimic real-world tool behavior, as well as realistic task graphs and gold-standard responses.


## Dataset Composition

The dataset is based on the **AsyncHow** dataset and has been extended to incorporate elements that enable detailed evaluation of agentic behavior, including parallel and sequential task graphs. It contains a diverse set of task graphs, tool functions, and expected outputs, allowing for comprehensive performance analysis.

### Key Components:
1. **Task Graphs**: Representations of tasks and their dependencies, capturing both parallel and sequential workflows.
2. **Tool Functions**: Synthetic Python functions generated to simulate real-world tools. These functions replicate various tasks, such as API call simulations or data retrieval.
3. **Expected Tool Call Sequences**: The ideal sequence in which tools should be called for successful task completion.
4. **Gold Standard Responses**: Benchmark outputs that represent the correct results for each task graph scenario.
5. **Complexity Categories**: Classification of task scenarios based on their structural complexity, such as linear workflows versus intricate interdependent tasks.

## Important assumptions we make
In order to use the dataset you will have to ensure that your agentic architecture allows you to retrieve the following data for each query:
1. **Task Graphs**: For each Task graph in the dataset you will have to obtain the task graph your system is producing
2. **List of tools**: For each query you will have to obtain the list of tools your agent is proposing
3. **Generated answer**: For each query, task graph and list of tools you will need to obtain the generated answer
4. **Tools**: You will have to ensure that your agent is using the tools that the were generated in `generate_functions.py`  the folders `tools_seq_50` and `tools_para_50`   
In the final step you will have to implement an evaluation function that loops through either `eval_data_para_df_50.csv` and `eval_data_seq_df_50.csv`, retrieves the data above
and merges the data into a `csv` or `pkl`.

## Before you get going
It is important to understand that we have adapted the AsyncHow graph data to the specific format we are using within our agentic system. 
We are using the following task graph structure currently:

        {
            "task_graph": {
                "nodes": [
                    {"id": "task_A", "label": "Task Description"},
                    {"id": "task_B", "label": "Task Description"},
                    {"id": "task_C", "label": "Task Description"},
                    {"id": "task_D", "label": "Task Description"}
                ],
                "edges": [
                    {"from": "task_A", "to": "task_B"},
                    {"from": "task_B", "to": "task_C"},
                    {"from": "task_B", "to": "task_D"},
                    {"from": "task_C", "to": "task_D"}
                ]
            }
        }

The structure we use is can be used for representing  parallel/sequential and asynchronous graph structures.
That means that all the functions in `transform_async_benchmark_data.py` are adapted to transform the graph structure from 
the AsyncHow dataset into the structure above. In order to use our dataset you will have to adapt the transformation functions
in `transform_async_benchmark_data.py` according to the graph structure you use in your agentic system. This task graphs
will be the baseline against which your agentic task graphs will be compared.

## Code for Dataset Generation

The dataset was generated using a Python script that leverages a GPT-based client to generate synthetic tool functions. These functions mimic real-world APIs, databases, or other tools that an agentic system may interact with. The steps involved in dataset creation include:
- **Tool Function Generation**: The `generate_functions` method creates Python functions based on tool descriptions and outputs them into the appropriate directories.
- **GPT-4 Function Creation**: A prompt is fed to the GPT-4 model, which generates Python functions and outputs based on provided tool descriptions. This approach ensures realistic, context-appropriate synthetic data generation.

### Key Code Snippets:

#### Generate Tool Functions
```python
def generate_functions(data, folder):
    base_directory = os.path.join("../..", folder)
    code = ""
    for function_name, function_code in zip(data["function_names"], data["functions"]):
        function_directory = os.path.join(base_directory, function_name)
        os.makedirs(function_directory, exist_ok=True)
        file_path = os.path.join(function_directory, f"{function_name}.py")
        with open(file_path, "w") as file:
            file.write(function_code)
            code = function_code
    return code
```

#### Example GPT-4 Prompt for Function Generation
```python
prompt = """
I am designing an evaluation dataset for an agentic system based on one or more Large Language Models (LLMs). 
Your task is to create Python functions that mimic potential tools in an agentic system...
"
```

The functions generated by this process are stored in the respective directories and are used in the evaluation of the agentic systems.

## Evaluation Metrics

Our dataset is paired with a set of evaluation metrics designed to assess the performance of agentic systems rigorously:

- **Node F1 Score**: Measures the system’s precision and recall in matching task nodes with the expected task graph.
- **Tool F1 Score**: Evaluates how accurately tools are selected and used in task execution.
- **Structural Similarity Index (SSI)**: Quantifies the overall fidelity of task graphs, capturing both node and edge similarities to ensure logical structure is preserved.
- **Node Label Similarity**: Measures the semantic similarity of nodes between the actual and expected task graphs using cosine similarity.
- **Graph Edit Distance (GED)**: A metric used to calculate the number of edits required to transform one graph into another.
You find the implementation of all the metrics in `metric_utils.py`.

## How to Use the Dataset

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/AsyncHow-Agentic-Systems-Evaluation.git
   cd AsyncHow-Agentic-Systems-Evaluation
   ```

2. **Task Graphs**: Adapt the transformation accordingly to the task graph format you are using
3. **Tool Functions**: Python scripts are provided for each scenario, replicating the synthetic tool functions. These can be executed and integrated into agentic systems for testing.
4. **Evaluation**: Use the provided metric scripts together with the data you obtained from you agentic system.

## License

This dataset is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Citation

If you use this dataset in your research, please cite our NeurIPS 2024 paper:

```bibtex
@inproceedings{anonymous2024advancing,
  title={Advancing Agentic Systems: Dynamic Task Decomposition, Tool Integration, and Evaluation using Novel Metrics and Dataset},
  author={Anonymous},
  booktitle={38th Conference on Neural Information Processing Systems (NeurIPS 2024)},
  year={2024}
}
```

## Reference

1. Lin, Fangru, La Malfa, Emanuele, Hofmann, Valentin, Yang, Elle Michelle, Cohn, Anthony G, and Pierrehumbert, Janet B. *Graph-enhanced Large Language Models in Asynchronous Plan Reasoning*. In Proceedings of the Forty-first International Conference on Machine Learning.
