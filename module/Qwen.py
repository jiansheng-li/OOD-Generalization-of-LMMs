import math
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_pretrained_qwen_model():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len



def eval_model(args):
    # Model
    tokenizer, model, context_len = load_pretrained_qwen_model()
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        image_path = os.path.join(args.image_folder, image_file)
        query = tokenizer.from_list_format([
            {'image': f'{image_path}'},
            {'text': f'{cur_prompt}'}
        ])
        with torch.inference_mode():
            outputs, history = model.chat(
                tokenizer=tokenizer,
                query=query,
                history=None
            )
        outputs = outputs.strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    args = parser.parse_args()
    eval_model(args)
