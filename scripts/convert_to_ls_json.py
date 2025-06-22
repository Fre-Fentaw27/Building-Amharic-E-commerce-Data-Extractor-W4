import json
from pathlib import Path

INPUT_JSONL_PATH = 'data/for_annotation/label_studio_tasks.jsonl'
OUTPUT_JSON_ARRAY_PATH = 'data/for_annotation/label_studio_tasks_array.json'

def convert_jsonl_to_json_array():
    tasks = []
    with open(INPUT_JSONL_PATH, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            if line.strip(): # Avoid empty lines
                tasks.append(json.loads(line))

    with open(OUTPUT_JSON_ARRAY_PATH, 'w', encoding='utf-8') as f_out:
        json.dump(tasks, f_out, indent=2, ensure_ascii=False)

    print(f"✅ Converted {len(tasks)} tasks from JSONL to JSON array: {OUTPUT_JSON_ARRAY_PATH}")

if __name__ == '__main__':
    convert_jsonl_to_json_array()