import json
import os
import torch
import argparse
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import open_clip
from open_clip.pretrained import download_pretrained_from_url
import sys
from evaluation.utils import setup_logging, analyse_unified_output
from data.random_sampler import gen_sample_json


def setup(args):
    # setup output directory and logging
    args.output_dir = f"{args.output_dir}/{args.model_name}/{args.dataset}/"
    os.makedirs(args.output_dir)
    logger = setup_logging(args.output_dir)
    logger.info(args)
    return logger


def main(args):
    model, _, preprocess = open_clip.create_model_and_transforms(f'{args.model_name}')
    tokenizer = open_clip.get_tokenizer(f'{args.model_name}')
    model.to('cuda')
    gen_sample_json(dataset=args.dataset, args=args)
    with open(f'{args.output_dir}/unified_input_{args.dataset}.json', 'r') as f:
        data = json.load(f)
    class_names = data['class_names']

    for item_id, item in tqdm(data['samples'].items(), desc="Processing Samples"):
        unified_output = {}
        image_path = os.path.join(
            args.data_dir, args.dataset, item['image'])
        try:
            image_item = Image.open(image_path).convert("RGB")
            image = preprocess(image_item).unsqueeze(0).to('cuda')
            text = tokenizer(class_names).to('cuda')
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                predicted_class_id = text_probs.argmax().item()
                predicted_class = class_names[predicted_class_id]
            unified_output['dataset'] = args.dataset
            unified_output['domain'] = item['domain']
            unified_output['subject'] = item['subject']
            unified_output['true_class'] = item['class']
            unified_output['predicted_class'] = predicted_class
            unified_output['image'] = item['image']
            unified_output['id'] = item_id
            with open(f'{args.output_dir}/unified_output_{args.model_name}.jsonl', 'a') as jsonl_file:
                jsonl_file.write(json.dumps(unified_output) + '\n')
        except (FileNotFoundError, IOError) as e:
            print(f"Error opening image {image_path}: {e}")
            continue
    logger.info("CLIPModel processes completed.")
    analyse_unified_output(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate CLIP in distribution shifts')
    parser.add_argument('--data_dir', type=str,
                        default="path to data dir")
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--output_dir', type=str, default="./exp_output")
    parser.add_argument('--num_sample', type=int, default=20,
                        help="the number of samples for each class")
    parser.add_argument('--model_name', type=str,
                        default="ViT-L-14", help='For more models, please refer to OpenAI')
    args = parser.parse_args()

    logger = setup(args)
    main(args)
