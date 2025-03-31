import argparse


def set_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LLaVA in distribution shifts')
    parser.add_argument('--data_dir', type=str, default=f'path to data dir')
    parser.add_argument('--output_dir', type=str, default="./exp_output")
    parser.add_argument('--first_num', type=int, default=0)
    parser.add_argument('--ice_num', type=str, default=0)
    parser.add_argument('--num_sample', type=int, default=500,
                        help="the number of samples for each class")
    parser.add_argument('--model_name', type=str,
                        default="gemini")
    parser.add_argument('--openai_api_key', type=str)
    parser.add_argument('--gemini_api_key', type=str)
    args = parser.parse_args()
    return args
