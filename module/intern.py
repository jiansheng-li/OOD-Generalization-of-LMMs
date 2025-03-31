import math
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from modelscope import AutoModel, AutoTokenizer


def load_pretrained_intern_model():
    tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-xcomposer-7b", trust_remote_code=True)
    model = AutoModel.from_pretrained("internlm/internlm-xcomposer-7b", device_map="cuda", trust_remote_code=True).cuda().eval()

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len


def eval_model(args):
    # Model
    tokenizer, model, context_len = load_pretrained_intern_model()
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
        with torch.inference_mode():
            outputs = model.generate(
                image=image_path,
                text=f'{cur_prompt}',
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=5000,
                use_cache=True,
                history=None)
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
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()
    eval_model(args)
