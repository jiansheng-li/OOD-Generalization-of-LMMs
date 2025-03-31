import json
import os
import torch
import argparse
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from evaluation.utils import setup_logging, analyse_unified_output
from data.generate_unified_input_data import gen_sample_json

def setup(args):
    # setup output directory and logging
    if args.output_dir[-6:] in 'exp_output':
        args.output_dir = f"{args.output_dir}/{args.model_name}/{args.dataset}"
        os.makedirs(args.output_dir)
    logger = setup_logging(args.output_dir)
    logger.info(args)
    return logger


def main(args):
    model = AutoModelForVision2Seq.from_pretrained("./model/kosmos/")
    processor = AutoProcessor.from_pretrained("./model/kosmos/")
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
            image = Image.open(image_path).convert("RGB")
            prompt = """Question: what is in this image? Answer :{answer}
            """
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_embeds=None,
                    image_embeds_position_mask=inputs["image_embeds_position_mask"],
                    use_cache=True,
                    max_new_tokens=1024,
                )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                processed_text, entities = processor.post_process_generation(generated_text)
                sentence=processed_text.lower().replace('woman','person').replace('man','person').replace('boy','person').replace('girl','person').replace('people','person')
                result='NULL'
                for i in class_names:
                    if i in sentence:
                        result=i
                        break
            unified_output['dataset'] = args.dataset
            unified_output['domain'] = item['domain']
            unified_output['subject'] = item['subject']
            unified_output['true_class'] = item['class']
            unified_output['predicted_class'] = result
            unified_output['image'] = item['image']
            unified_output['id'] = item_id
            with open(f'{args.output_dir}/unified_output_{args.model_name}.jsonl', 'a') as jsonl_file:
                jsonl_file.write(json.dumps(unified_output) + '\n')
        except Exception as e:
            print(e)
        analyse_unified_output(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate CLIP in distribution shifts')
    parser.add_argument('--data_dir', type=str,
                        default="path to data dir")
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--output_dir', type=str, default="./exp_output")
    parser.add_argument('--num_sample', type=int, default=500,
                        help="the number of samples for each class")
    parser.add_argument('--model_name', type=str,
                        default="kosmos")
    args = parser.parse_args()

    logger = setup(args)
    main(args)
