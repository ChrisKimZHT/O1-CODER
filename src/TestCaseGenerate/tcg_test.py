import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging,
    set_seed,
)


def template(problem: str, tag: list, code: str) -> str:
    return f'''### Instruction
Please complete the task in the code part and generate some test case in the test part that can be used to test the quality of the generated code.
### Problem
{problem}
### Code Part
{", ".join(tag[:20])}
```python
{code}
```
### Test Part
[Generate 3 test cases here to validate the code.]'''


def main():
    logging.set_verbosity_warning()
    set_seed(42)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    taco_dataset = load_dataset(args.taco_path)
    taco_test = taco_dataset["test"].shuffle(seed=42)
    taco_test = taco_test.select(range(10))

    for element in taco_test:
        problem = element["question"]
        tag = eval(element["tags"])
        code = eval(element["solutions"])[0]

        input_text = template(problem, tag, code)
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

        print(input_text)

        outputs = model.generate(
            inputs,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            num_return_sequences=1,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(generated_text)
        print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="/home/chriskim/O1-CODER/src/TestCaseGenerate/finetune_deepseek1.3_instruct_o1_format_SFT/checkpoint-3000")
    parser.add_argument("--taco-path", type=str, default="/home/chriskim/TACO")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    main()
