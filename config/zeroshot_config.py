import argparse


def set_args():
    parser = argparse.ArgumentParser(
        description='Evaluate zeroshot for each model in distribution shifts')
    parser.add_argument('--data_dir', type=str,
                        default="./data_dir", help="path to data_dir")
    parser.add_argument('--dataset', type=str, nargs='+', default=["VLCS"])
    parser.add_argument('--output_dir', type=str, default="./exp_output")
    parser.add_argument('--num_sample', type=int, default=10,
                        help="the number of samples for each class")
    parser.add_argument('--model_name', type=str,
                        choices=['LLaVA', 'Qwen', 'mPlug', 'intern', 'cogvlm', 'minigpt', 'LLaMA', 'blip2',
                                 'instructblip', 'emu', 'gpt', 'gemini'],
                        default="LLaMA", help='choose a model to test')
    parser.add_argument('--openai_api_key', type=str)
    parser.add_argument('--gemini_api_key', type=str)
    args = parser.parse_args()
    return args
