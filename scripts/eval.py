import os
import json
import argparse
from eval_utils import evaluate

def main():
    parser = argparse.ArgumentParser(description='evaluation of LLM-TPC')
    parser.add_argument('--log_dir', type=str, help='path to log files')
    args = parser.parse_args()
    
    predictions = []
    for file in os.listdir(args.log_dir):
        pred = json.load(open(os.path.join(args.log_dir, file), "r"))[-1]
        if "answer" in pred:
            predictions.append(pred)

    metric = "soft match"
    acc = evaluate(predictions, metric)
    print(f"{metric} accuracy of {len(predictions)} samples:", '{:.2%}'.format(acc))


if __name__ == "__main__":
    main()