## How to Use

### Data Preparation

You can find the TACO dataset on [Hugging Face](https://huggingface.co/datasets/BAAI/TACO). Convert the dataset into a JSON file and place it in the `\data\TACO` directory.

---

### Generate Data

#### Local Model

Run the `run_TACO.sh` script with the appropriate configurations to use the local model for data generation:

```bash
bash run_TACO.sh
```

#### Main Arguments

| Argument              | Type   | Description                             |
|-----------------------|--------|-----------------------------------------|
| `--dataset_name`      | str    | Name of the dataset folder in the `data` directory |
| `--test_json_filename`| str    | Name of the JSON file containing the data |
| `--model_ckpt`        | str    | Path to the model checkpoint            |
| `--num_rollouts`      | int    | Number of MCTS rollouts                 |
| `--max_depth_allowed` | int    | Maximum depth allowed for the MCTS search tree |

---

#### OpenAI API

First, configure your `api_key` in the `\models\OpenAI_API.py` file. 

Then, run the `api_run_TACO.sh` script to use the API for data generation:

```bash
bash api_run_TACO.sh
```

#### Additional Arguments

| Argument              | Type   | Description                             |
|-----------------------|--------|-----------------------------------------|
| `--api`               | str    | Default is `vllm`, which calls the local model |
| `--model_ckpt`        | str    | Specific OpenAI model name              |

--- 

### Data Example

A sample dataset can be found in the `run_outputs` folder. Detailed information for each problem is available in the `answer_sheets` folder.

#### File Information

| Filename                                | Description                             |
|-----------------------------------------|-----------------------------------------|
| `Question XXXX - Answer.json`           | Contains the original question information |
| `Question XXXX - Best Solution.json`    | The path with the highest reward in the final step |
| `Question XXXX - Complete Solutions.json` | All complete paths in the MCTS search tree |
| `Question XXXX - Rollout Solutions.json` | Paths generated during each MCTS rollout |
| `args.json`                             | Parameter configuration information     |
| `intermediate_result.txt`               | Logs for model calls and intermediate results |


## Acknowledge

This code is derived from and modified based on the project available at [https://github.com/zhentingqi/rStar/](https://github.com/zhentingqi/rStar/).
