import google.generativeai as genai
from tqdm import tqdm
from PIL import Image
from data.generate_unified_input_data import *
from data.generate_unified_output_data import *
from config.ICL_config import set_args
from evaluation.utils import setup_logging,analyse_unified_output


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


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
        genai.configure(api_key=f'{args.gemini_api_key}', transport='rest')
        model = genai.GenerativeModel('gemini-pro-vision')
        ice_data_path = dataset
        for line in tqdm(questions):
            idx = line["question_id"]
            image_file = line["image"]
            qs = line["text"]
            qs = qs.replace('Given the image', 'Given the image above')
            qs = qs.replace('in this image', 'in the image above')
            ans = '''  ---BEGIN FORMAT TEMPLATE---
                               Answer Choice: [Your Answer Choice Here]
                               ---END FORMAT TEMPLATE---'''

            ice_num = args.ice_num
            content = []
            res_qs = ''
            for i in range(ice_num):
                j = int(i / 2) if i % 2 == 0 else int(len(ice_data_path) - 1 - i // 2)
                image_path = ice_data_path[j]
                label = image_path.split('\\')[-2]
                image = Image.open(image_path).convert('RGB')
                image = image.resize((768, 768))
                ice_text = qs + ans.replace('[Your Answer Choice Here]', label) + '\n\n'
                content.append(image)
                content.append(ice_text)
            image = Image.open(image_file).convert('RGB')
            res_qs += str(image_file)
            image = image.resize((768, 768))
            content.append(image)
            content.append(qs)
            res_qs += qs
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
