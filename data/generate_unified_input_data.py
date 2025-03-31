import os
import json
import logging


def gen_sample_json(dataset='PACS', args=None):
    logger = logging.getLogger('MLLMs')
    logger.info(
        f'sampling data examples in {dataset}, and writing into unified_input_{dataset}.json')
    # Load the JSON file
    with open('./data/dataset_info.json', 'r') as f:
        data = json.load(f)
        dataset_info = data[dataset]
    domains = dataset_info['domains']
    class_names = dataset_info['class_names']
    subject = dataset_info['subject']
    logger.info('Processing {}: {} domains, {} classes'.format(
        dataset, len(domains), len(class_names)))
    selected_images_info = {'dataset': dataset, 'domains': domains,
                            'class_names': class_names, 'samples': {}}

    for domain in domains:
        domain_path = os.path.join(args.data_dir, dataset, domain)
        if os.path.exists(domain_path) and os.path.isdir(domain_path):
            for class_id, class_name in enumerate(selected_images_info['class_names']):
                class_path = os.path.join(domain_path, class_name)

                if os.path.exists(class_path) and os.path.isdir(class_path):
                    images = [os.path.join(domain, class_name, img) for img in os.listdir(
                        class_path) if img.endswith((".jpg", ".png", ".jpeg"))]

                    if len(images) >= args.num_sample:
                        sampled_images = images[0:args.num_sample]
                    else:
                        sampled_images = images
                    for image in sampled_images:
                        image_id = len(selected_images_info['samples']) + 1
                        selected_images_info['samples'][str(image_id)] = {
                            "domain": domain,
                            "class": class_name,
                            "image": image,
                            "class_id": str(class_id),
                            "subject": subject
                        }
                else:
                    logger.info(
                        f"The class {class_name} does not exist in {domain}.")

    with open(f'{args.output_dir}/unified_input_{dataset}.json', 'w') as f:
        json.dump(selected_images_info, f, indent=4)


def convert_unified_input_into_zeroshot_vqa(dataset, data, args):
    question_file = f'{args.output_dir}/input_{dataset}_in_{args.model_name}_vqa.jsonl'
    first_flag = True
    class_names = data['class_names']
    for item_id, item in data['samples'].items():
        example_vqa_format = {}
        example_vqa_format['image'] = os.path.join(dataset, item['image'])
        prompt = f"""Given the image, answer the following question using the specified format.
        Question: What is in this image? Choice list: {class_names}.
        Please choose a choice from the list and respond with the following format:
        ---BEGIN FORMAT TEMPLATE---
        Answer Choice: [Your Answer Choice Here]
        ---END FORMAT TEMPLATE---
        Do not deviate from the above format. Repeat the format template for the answer.
        """
        example_vqa_format['text'] = prompt
        example_vqa_format['subject'] = item['subject']
        example_vqa_format['question_id'] = item_id
        mode = 'w' if first_flag else 'a'
        with open(question_file, mode) as jsonl_file:
            jsonl_file.write(json.dumps(example_vqa_format) + '\n')
        first_flag = False
    return question_file
