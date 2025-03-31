import base64
import requests
import json
from tqdm import tqdm
from data.generate_unified_input_data import *
from data.generate_unified_output_data import *
from config.ICL_config import set_args
from evaluation.utils import setup_logging, analyse_unified_output


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')


def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]



def main(args):
    for idx, each_dataset in enumerate(args.dataset):
        gen_sample_json(dataset=each_dataset, args=args)
        # Load the JSON file
        with open(f'{args.output_dir}/unified_input_{each_dataset}.json', 'r') as f:
            data = json.load(f)
        question_file = convert_unified_input_into_zeroshot_vqa(
            each_dataset, data, args)
        answers_file = f"{args.output_dir}/output_{each_dataset}_in_genai_vqa.jsonl"
        questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")
        first_dataset_flag = True if idx == 0 else False
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.openai_api_key}"
        }
        for line in tqdm(questions):
            idx = line["question_id"]
            image_file = line["image"]
            qs = line["text"]
            cur_prompt = qs
            base64_img = image_to_base64(os.path.join(args.data_dir, args.dataset[0], image_file))
            prompt_messages = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": qs
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        },
                        "resize": 768
                    }
                ],
            }
            parm = {
                'model': "gpt-4-vision-preview",
                'messages': [prompt_messages],
                'max_tokens': 1024}
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions", headers=headers, json=parm).json()
                ans_file.write(json.dumps({"question_id": idx,
                                           "prompt": cur_prompt,
                                           "text": response['choices'][0]['message']['content'],
                                           "metadata": {}}) + "\n")
                ans_file.flush()
            except Exception as e:
                print(f"error: {type(e).__name__}")
                print(f"error message: {e}")
        ans_file.close()

        # Run the subprocess and capture the output
        # Log stdout
        convert_zeroshot_answer_into_unified_output(
            dataset=each_dataset, answer_file=answers_file, unified_input=data, first_dataset_flag=first_dataset_flag)
    analyse_unified_output(args)


if __name__ == '__main__':
    args = set_args()
    main(args)
