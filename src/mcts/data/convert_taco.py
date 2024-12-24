import argparse
import json
import random
import tempfile
from datasets import load_dataset


def main():
    taco = load_dataset(args.input_taco)
    taco_test = taco[args.split]
    with tempfile.NamedTemporaryFile(mode="r+") as f:
        taco_test.to_json(f.name)
        lines = f.readlines()
    result = []
    for line in lines:
        element = json.loads(line)
        element["input_output"] = json.loads(element["input_output"])
        result.append(element)
    example = random.choice(result)
    print(json.dumps(example, indent=2))
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-taco", type=str, default="/home/chriskim/TACO")
    parser.add_argument("--output-json", type=str, default="/home/chriskim/taco.json")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    main()
