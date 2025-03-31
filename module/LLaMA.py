import argparse
import os
import json
from tqdm import tqdm
import shortuuid
import cv2
from PIL import Image
import math
import llama



def eval_model(args):
    # Model
    llama_dir = 'your path to llama model'
    model, preprocess = llama.load("BIAS-7B", llama_dir, llama_type="7B", device='cuda')
    model.eval()
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        prompt = llama.format_prompt(qs)
        try:
            img = Image.fromarray(cv2.imread(image_file))
            img = preprocess(img).unsqueeze(0).to('cuda')
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                       "prompt": cur_prompt,
                                       "text": model.generate(img, [prompt])[0],
                                       "answer_id": ans_id,
                                       "model_id": 'llama',
                                       "metadata": {}}) + "\n")
            ans_file.flush()
        except Exception as e:
            continue
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
