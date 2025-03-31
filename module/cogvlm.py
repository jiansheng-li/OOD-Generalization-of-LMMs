import math
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

def load_pretrained_cogvlm_model():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    # load model
    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to('cuda').eval()
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model,context_len




def eval_model(args):
    # Model
    tokenizer, model, context_len = load_pretrained_cogvlm_model()
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        history = None
        torch_type = torch.bfloat16
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        input_by_model = model.build_conversation_input_ids(tokenizer, query=qs, history=history, images=[image],template_version='vqa')
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
        }

        gen_kwargs = {"max_length": 2048,
                      "temperature": 0.9,
                      "do_sample": False}

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)
            output_ids = output_ids[:, inputs['input_ids'].shape[1]:]
            outputs = tokenizer.decode(output_ids[0])
            outputs = outputs.split("</s>")[0]
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
