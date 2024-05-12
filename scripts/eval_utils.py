import re

def is_synonym(answer, gt_answer):
    for key, value in synonym.items():
        if gt_answer in value:
            if answer in value:
                return True
            for candidate in value:
                if candidate in answer:
                    return True
    return False

def clean_answer(data):
    if data is None:
        # print(1)
        return ""    
    data = data.lower()
    data = re.sub('[^a-zA-Z0-9]+', ' ', data)
    data = re.sub(' {2,}', ' ', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)
    return data

synonym = {
    "left": ["left", "7 o'clock", "8 o'clock", "9 o'clock", "10 o'clock", "11 o'clock"],
    "right": ["right", "1 o'clock", "2 o'clock", "3 o'clock", "4 o'clock", "5 o'clock"],
    "front": ["front", "forward", "forwards", "in front", "infront", "10 o'clock", "11 o'clock", "12 o'clock", "1 o'clock", "2 o'clock"],
    "behind": ["behind", "back", "backward", "backwards", "4 o'clock", "5 o'clock", "6 o'clock", "7 o'clock", "8 o'clock"],
    "true": ["true", "yes"],
    "false": ["false", "no"],

    "big": ["big", "large"],
    "circle": ["circle", "circular", "oval", "round"],
    "rectangle": ["rectangle", "rectangular"],

    "box": ["box", "boxes"],
    "cabinet": ["cabinet", "cabinets"],
    "chair": ["chair", "chairs"],
    "clothes dryer": ["clothes dryer", "clothes dryers"],
    "clothing": ["clothing", "clothes"],
    "cube": ["cube", "cubes"],
    "curtain": ["curtain", "curtains"],
    "divider": ["divider", "dividers"],
    "dryer": ["dryer", "dryers"],
    "kitchen cabinet": ["kitchen cabinet", "kitchen cabinets"],
    "mail box": ["mail box", "mail boxes"],
    "mini fridge": ["minifridge", "mini fridge"],
    "monitor": ["monitor", "monitors"],
    "picture": ["picture", "pictures"],
    "pillow": ["pillow", "pillows"],
    "pipe": ["pipe", "pipes"],
    "plant": ["plant", "plants"],
    "rack": ["rack", "rack stand"],
    "towel": ["towel", "towels"],
    "trash bin": ["trash bin", "trash bins", "trash can", "trashcan"],
    "washing machine": ["washing machine", "washing machines"],
    "whiteboard": ["whiteboard", "white board"],
    "window": ["window", "windows"],
}

for key,value in synonym.items():
    for i, candidate in enumerate(value):
        synonym[key][i] = clean_answer(candidate)

all_synonym = {}
for key in synonym:
    for val in synonym[key]:
        all_synonym[val] = synonym[key]

def soft_match(answer, gt_answer):
    answer = clean_answer(answer)
    gt_answer = clean_answer(gt_answer)
    if answer == gt_answer:
        return True
    elif answer in gt_answer:
        return True
    elif ''.join(answer.split()) in ''.join(gt_answer.split()):
        return True
    elif ''.join(gt_answer.split()) in ''.join(answer.split()):
        return True
    elif len(set(answer.split()).intersection(gt_answer.split())) > 0:
        return True
    elif is_synonym(answer, gt_answer):
        return True
    return False

def strict_match(answer, gt_answer):
    answer = clean_answer(answer)
    gt_answer = clean_answer(gt_answer)
    if answer == gt_answer:
        return True
    return False

def evaluate(predictions, metric="soft match"):
    assert metric in ["soft match", "strict match", "strict match with synonym"]
    eval_metric_dict = {
        "soft match": soft_match,
        "strict match": strict_match
    }
    acc = 0
    for pred in predictions:
        pred_answer = pred["answer"]
        gt_answer = pred["gt_answer"]
        if eval_metric_dict[metric](pred_answer, gt_answer):
            acc += 1
    return acc/len(predictions)
