import subprocess
import os
import json
from datetime import datetime
from evaluation.utils import setup_logging, analyse_unified_output
from data.generate_unified_input_data import gen_sample_json, convert_unified_input_into_zeroshot_vqa
from data.generate_unified_output_data import convert_zeroshot_answer_into_unified_output
from config.zeroshot_config import set_args


def setup(args):
    # setup output directory and logging
    current_time = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
    args.output_dir = f"{args.output_dir}/{current_time}/{args.model_name}/{args.dataset[0]}"
    os.makedirs(args.output_dir)
    logger = setup_logging(args.output_dir)
    logger.info('Start a new experiment')
    # Set up logging
    logger.info(args)
    return logger


def main(args):
    logger = setup(args)
    for idx, each_dataset in enumerate(args.dataset):
        gen_sample_json(dataset=each_dataset, args=args)
        # Load the JSON file
        with open(f'{args.output_dir}/unified_input_{each_dataset}.json', 'r') as f:
            data = json.load(f)
        logger.info(
            'Convert the unified input format into zeroshot vqa format')
        question_file = convert_unified_input_into_zeroshot_vqa(
            each_dataset, data, args)
        answer_file = f"{args.output_dir}/output_{each_dataset}_in_zeroshot_vqa.jsonl"
        first_dataset_flag = True if idx == 0 else False
        zeroshot_model_vqa = [
            "python", f"module/{args.model_name}.py",
            "--question-file", question_file,
            "--image-folder", f"{args.data_dir}",
            "--answers-file", answer_file
        ]
        # Run the subprocess and capture the output
        result = subprocess.run(
            zeroshot_model_vqa, stdout=subprocess.PIPE, text=True)
        # Log stdout
        if result.stdout:
            logger.info("zeroshot Output:\n" + result.stdout)
        convert_zeroshot_answer_into_unified_output(
            dataset=each_dataset, answer_file=answer_file, unified_input=data, first_dataset_flag=first_dataset_flag,
            args=args)
    analyse_unified_output(args)


if __name__ == '__main__':
    args = set_args()
    main(args)
