import json

input_file = 'data/SMSSpamCollection'

output_file = 'data/spam_classification.jsonl'

instruction_variations = [
    "Classify the following SMS message as 'ham' or 'spam'.",
    "Determine whether this message is spam or not:",
    "Is the following message spam or ham?",
    "Spam or ham? Classify this SMS.",
    "Does this message contain spam?"
]

def convert_to_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                label, message = line.split('\t', 1)
            except ValueError:
                print(f"Skipping malformed line: {line}")
                continue

            instruction = instruction_variations[hash(line) % len(instruction_variations)]

            json_obj = {
                "instruction": instruction,
                "input": message.strip(),
                "output": label.strip().lower()
            }

            outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

    print(f"Conversion complete! Saved to {output_path}")

convert_to_jsonl(input_file, output_file)
