import argparse
import math
from tqdm import tqdm
import google.generativeai as genai
from PIL import Image
from config.zeroshot_config import set_args
from data.generate_unified_output_data import *
from data.generate_unified_input_data import *
from evaluation.utils import setup_logging, analyse_unified_output


def setup(args):
    # setup output directory and logging
    args.output_dir = f"{args.output_dir}/{args.model_name}/{args.dataset}/"
    os.makedirs(args.output_dir)
    logger = setup_logging(args.output_dir)
    logger.info(args)
    return logger


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


def main(args):
    logger = setup(args)
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
        genai.configure(api_key=f'{args.gemini_api_key}', transport='rest')
        model = genai.GenerativeModel('gemini-pro-vision')
        for line in tqdm(questions):
            idx = line["question_id"]
            image_file = line["image"]
            qs = line["text"]
            content = []
            image = Image.open(image_file).convert('RGB')
            image = image.resize((768, 768))
            content.append(image)
            content.append(qs)
            try:
                response = model.generate_content(content, stream=False)
                response.resolve()
                ans_file.write(json.dumps({"question_id": idx,
                                           "prompt": res_qs,
                                           "text": response.text,
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
