import base64
import requests
import math
from tqdm import tqdm
from data.generate_unified_input_data import *
from data.generate_unified_output_data import *
from config.ICL_config import set_args
from evaluation.utils import setup_logging, analyse_unified_output


def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')


def gen_prompt(ice_num, qs, line, query_img_base64, ice_data_path):
    # qs = qs.split('Please respond')[0].split('format. \n')[-1].split('\n')[0]
    qs = qs.replace('Given the image', 'Given the image above')
    qs = qs.replace('in this image', 'in the image above')

    ans = '''---BEGIN FORMAT TEMPLATE---
        Answer Choice: [Your Answer Choice Here]
        ---END FORMAT TEMPLATE---'''
    prompt_messages = {
        "role": "user",
        "content": [],
    }

    for i in range(ice_num):
        j = int(i / 2) if i % 2 == 0 else int(len(ice_data_path) - 1 - i // 2)

        image_path = ice_data_path[j]
        print('image_path', image_path)
        print(ice_data_path)
        prompt_messages['content'].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_to_base64(image_path)}"
            },
            "resize": 768
        })

        label = image_path.split('/')[-2]
        # ice_text = ans.replace('[Your Answer Choice Here]', label) + '\n\n'
        ice_text = qs + ans.replace('[Your Answer Choice Here]', label) + '\n\n\n'
        prompt_messages['content'].append({
            "type": "text",
            "text": ice_text
        }, )

    # for the query image
    prompt_messages['content'].append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{query_img_base64}"
        },
        "resize": 768
    })
    qs_text = qs + 'Do not response with I\'m sorry. Just pick a choice.\n'
    prompt_messages['content'].append({
        "type": "text",
        "text": qs_text
    }, )
    return prompt_messages


def main(args, dataset):
    for idx, each_dataset in enumerate(args.dataset):
        gen_sample_json(dataset=each_dataset, args=args)
        # Load the JSON file
        with open(f'{args.output_dir}/unified_input_{each_dataset}.json', 'r') as f:
            data = json.load(f)
        question_file = convert_unified_input_into_zeroshot_vqa(
            each_dataset, data, args)
        answers_file = f"{args.output_dir}/output_{each_dataset}_in_genai_vqa.jsonl"
        questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
        questions = get_chunk(questions, 1, 0)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")
        first_dataset_flag = True if idx == 0 else False
        ice_data_path = dataset
        for line in tqdm(questions):
            idx = line["question_id"]
            image_file = line["image"]
            qs = line["text"]
            cur_prompt = qs
            base64_img = image_to_base64(image_file)
            prompt_messages = gen_prompt(ice_num=args.ice_num, qs=qs, line=line, query_img_base64=base64_img,
                                         ice_data_path=ice_data_path)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {args.openai_api_key}"
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
    args.output_dir = f"{args.output_dir}/{args.model_name}/{args.dataset}/{args.ice_num}"
    os.makedirs(args.output_dir)
    logger = setup_logging(args.output_dir)
    logger.info(args)
    with open('./data/dataset_info_icl.json', 'r') as f:
        data = json.load(f)
    for j in ['NIH-Chest', 'XCOVFour', 'CT-Xcov']:
        dataset = []
        args.dataset = [j]
        ice_data_dir = f'{args.data_dir}/{args.dataset[0]}_ICE'
        if not os.path.exists(ice_data_dir):
            continue
        for root, dirs, files in os.walk(ice_data_dir):
            for file in files:
                if file.endswith('jpeg') or file.endswith('jpg') or file.endswith('png'):
                    path = os.path.join(root, file)
                    dataset.append(path)
        main(args, dataset)
