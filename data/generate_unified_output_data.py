import json
import re


def search_pred_info(answer_by_zeroshot, class_names):
    text_in_zeroshot = answer_by_zeroshot['text']
    predicted_class = None
    for class_name in class_names:
        # Pattern to match 'Answer Choice: [class_name]' or 'Answer Choice: class_name' (case-insensitive)
        pattern = re.compile(
            r"Answer Choice:\s*(?:\[)?'?\"?" +
            re.escape(class_name) + r"'?\"?(?:\])?",
            re.IGNORECASE
        )
        if pattern.search(text_in_zeroshot):
            predicted_class = class_name
            break
    # Regular expression patterns to extract Confidence Score (0~1) and Reasoning
    confidence_score_pattern = r'Confidence Score:\s*([0-9]*\.?[0-9]+)'
    reasoning_pattern = r'Reasoning:\s*(.+)'

    # Extract Confidence Score
    confidence_score_match = re.search(
        confidence_score_pattern, text_in_zeroshot, re.DOTALL)
    if confidence_score_match:
        confidence_score = confidence_score_match.group(1).strip()
    else:
        confidence_score = None

    # Extract Reasoning
    reasoning_match = re.search(reasoning_pattern, text_in_zeroshot, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        reasoning = None
    return predicted_class, confidence_score, reasoning


def convert_zeroshot_answer_into_unified_output(dataset, answer_file, unified_input, args, first_dataset_flag=True):
    with open(answer_file, 'r') as file:
        class_names = unified_input['class_names']
        mode = 'w' if first_dataset_flag else 'a'
        for line in file:
            answer_by_zeroshot = json.loads(line)
            item_id = answer_by_zeroshot['question_id']
            item = unified_input['samples'][item_id]
            unified_output = {}
            # Store unified output jsonl
            unified_output['dataset'] = dataset
            unified_output['domain'] = item['domain']
            unified_output['subject'] = item['subject']
            unified_output['true_class'] = item['class']
            predicted_class, confidence_score, reasoning = search_pred_info(
                answer_by_zeroshot, class_names)
            unified_output['predicted_class'] = predicted_class
            unified_output['image'] = item['image']
            unified_output['id'] = item_id
            unified_output['confidence_score'] = confidence_score
            unified_output['reasoning'] = reasoning
            with open(f'{args.output_dir}/unified_output_{args.model_name}.jsonl', mode) as jsonl_file:
                jsonl_file.write(json.dumps(unified_output) + '\n')
            mode = 'a'
