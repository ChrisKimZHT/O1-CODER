## How to use
To initialize the environment, you need to install the required packages. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies listed in the `requirements.txt` file.

### Data
We provide data examples for PRM training in the `data` folder. The reward labels are available in both hard and soft estimation forms.

For the hard estimation labels, you can refer to `data/examples/hard_label_examples.json` for processing, while the corresponding soft label forms are provided in `data/examples/soft_label_examples.json`.

### Train
#### Basic Usage
Ensure the path is within the `prm_training` folder and run the following script
```bash
bash run.sh
```
#### Main Arguments
|               |        |                                   |
|---------------|--------|-----------------------------------|
| `--config_file` | str | accelerate config file path |
| `--model_name_or_path` | str | your model path |
| `--data_path` | str | data for training |
| `--use_soft_label` | bool | Whether to use soft labels during training, default is false |