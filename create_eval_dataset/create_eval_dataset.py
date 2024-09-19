from gpt.gpt_client import GPT4oAzureClientManager
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer, util
from loguru import logger


def remove_semantic_duplicates(folder_path, similarity_threshold=0.8):

    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Collect all subfolder names
    subfolder_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    # Compute embeddings for all subfolder names
    embeddings = model.encode(subfolder_names, convert_to_tensor=True)

    # Identify and remove semantically similar duplicates
    unique_subfolders = []
    for i, name in enumerate(subfolder_names):
        if not any(util.cos_sim(embeddings[i], embeddings[j]) > similarity_threshold for j in range(i)):
            unique_subfolders.append(name)

    return unique_subfolders


def generate_functions(data,folder):
    # Define the base directory where the "tools" folder is located
    base_directory = os.path.join("../..", folder)
    code = ""

    # Iterate over the function names and corresponding function code
    for function_name, function_code in zip(data["function_names"], data["functions"]):
        # Create a directory within the "tools" folder with the function name
        function_directory = os.path.join(base_directory, function_name)
        os.makedirs(function_directory, exist_ok=True)

        # Create a .py file within the directory
        file_path = os.path.join(function_directory, f"{function_name}.py")
        with open(file_path, "w") as file:
            file.write(function_code)
            code = function_code

    return code

def repair_json(response):
    try:
        response = json.loads(response.replace("```json", "").replace("```", ""))
    except Exception as e:
        print()
    return response


def generate_functions_from_tool_list():
    prompt = """ 
    I am designing an evaluation dataset for an agentic system based on one or more Large Language Models (LLMs). 
Your task is to create Python functions that mimic potential tools in an agentic system. Create only a single function for each item in a list of tool descriptions and base the signature and function body only on the description.
Generate synthetic data that each function must always return as a string, closely mimicking the type of data indicated by the function name and ensuring it is as realistic as possible. 
This data should resemble typical outputs from sources such as APIs, databases, spreadsheets, or factual information.
Given the following list of tools, create the minimum number of functions required to handle all possible subtasks, ensuring each function is aligned with a specific tool and executed according to its position in the list.

### Instructions:

1. **Function Creation:**
   - For each tool description in the provided list, create a Python function that mimics the tool's functionality.
   - The function signature and body should be based solely on the provided tool description.
   - Each function should generate synthetic data that is always returned as a string, closely mimicking the type of data suggested by the function name.
   - This synthetic data should be realistic and representative of typical outputs from sources such as APIs, databases, spreadsheets, or other factual information.
   - Include any necessary imports (e.g., `import json`) within the function to handle data generation or formatting.
   - Do not use types in the function signature neither for the input parameters nor the output parameters
   - Do not use the input paramters in the return of the function
   - Each function must contain a detailed Google-style docstring, adhering strictly to the following format:

    \"""
    Detailed description of the what the function does

    Args:
        input_1(str): Detailed description of the first input parameter.
        input_2 (bool): Detailed description of the second input parameter.

    Returns:
        str: Detailed explanation of what the function returns.
    \"""

2. **Function Execution and Response Generation:**
   - Execute the functions based on their position in the provided list of tools.
   - Use the outputs from these functions to formulate a final response that directly answers the user query.
   - The response should be accurate, comprehensive, and make effective use of the data returned by the functions.

### Output Requirements:
- The final output should be a valid JSON object with three keys:
  - `"functions"`: A list of the Python functions you have created, each represented as a string of valid Python code.
  - `"function_names"`: A list of the generated function names.
  - `"gold_standard_response"`: A precise, short, and well-structured response that answers the user query using the tools.
  - `"complexity_score"`: categorize each query into one of these categories: "Food and Entertainment", "Education and Communication", "Family Life", "Home and Garden", "Health", "Others"

### Provided Inputs:
- **Tool Descriptions**: {tools}
- **User Query**: {query}

### Note:
- Ensure that the functions are closely aligned with the provided tool descriptions and generate realistic data that reflects the expected outputs.
- Only output the JSON with the three keys! This JSON will be automatically parsed, so ensure the format is precise.
"""

#    eval_df = pd.read_csv("../asynchow_dataset/asynchow_seq_df.csv")
    eval_df = pd.read_csv("../asynchow_dataset/asynchow_para_df.csv")
    #tool_folder = "tools_seq_50"
    tool_folder = "tools_para_50"
    target_df = "eval_data_para_df_50.csv"

    tool_lists = eval_df["Tools"].to_list()
    user_queries = eval_df["Scenario Name"].to_list()
    task_graphs = eval_df["Task Graphs"].to_list()

    my_gpt_client = GPT4oAzureClientManager()

    expected_tools = []
    expected_responses = []
    complexity_scores = []
    selection_idx = 51
    codes = []

    failed_entries_num = 0

    for idx, (tool_list, query) in enumerate(zip(tool_lists[:selection_idx], user_queries[:selection_idx])):
        response = my_gpt_client.query_gpt_4o(prompt.format(tools=tool_list, query=query))
        response = repair_json(response)
        logger.debug("CURRENT IDX" + str(idx))
        logger.debug("CURRENT Query" + str(query))
        try:
            expected_tools.append([response["function_names"]])
            expected_responses.append([response["gold_standard_response"]])
            complexity_scores.append([response["complexity_score"]])
            code = generate_functions(response, folder=tool_folder)
            codes.append(code)
        except:
            logger.debug("Skip extraction" + str(response["function_names"]))
            failed_entries_num+=1
            continue


    dataset = {"Scenario Name" : user_queries[:len(complexity_scores)],
               "Category" : complexity_scores,
               "Tools" : tool_lists[:len(complexity_scores)],
               "expected_tool_calls" : expected_tools,
               "Task Graph" : task_graphs[:len(complexity_scores)],
               "gold_standard_response" : expected_responses,
               "codes" : codes
    }
    new_metrics_df = pd.DataFrame(dataset)
    new_metrics_df.to_csv(target_df)


if __name__ == "__main__":
    #generate_functions_from_tool_list()

    eval_df = pd.read_csv("../asynchow_dataset/asynchow_para_df.csv")
    task_graphs = eval_df["Task Graphs"].to_list()

    target_df = pd.read_csv("eval_data_para_df_50.csv")
    target_df["Task Graphs"] = task_graphs
    #target_df.to_csv("eval_data_para_df_50.csv")