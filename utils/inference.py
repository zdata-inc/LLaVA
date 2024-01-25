import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import textwrap
import random
from tqdm import tqdm
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from worker import eval_model, load_model
from llava.mm_utils import get_model_name_from_path

random_seed = 42  # You can use any integer as the seed
random.seed(random_seed)

model_path = "/home/asakhare/mnt/llava/models/liuhaotian/llava-v1.5-13b"
data_dir = '/home/asakhare/mnt/solink/data'
output_dir = '/home/asakhare/mnt/solink/output'

prompts = ["Describe the image concisely.",
           "Provide a brief description of the given image.",
           "Offer a succinct explanation of the picture presented.",
           "Summarize the visual content of the image.",
           "Give a short and clear explanation of the subsequent image.",
           "Share a concise interpretation of the image provided.",
           "Present a compact description of the photo’s key features.",
           "Relay a brief, clear account of the picture shown.",
           "Render a clear and concise summary of the photo.",
           "Write a terse but informative summary of the picture.",
           "Create a compact narrative representing the image presented."
           ]

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "conv_mode": None,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

tokenizer, model, image_processor, context_len = load_model(args)

results = []

for split in ['train', 'test']:
    df = pd.read_csv(os.path.join(data_dir, split/ split + '.csv'))
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing events"):
        dataset_id = row['dataset_id']
        event_id = row['event_id']
        image_id = row['event_id']

        image_path = os.path.join(data_dir, split, dataset_id, event_id, image_id)

        args.image_file = image_path
        args.model = model
        args.tokenizer = tokenizer
        args.context_len = context_len
        args.image_processor = image_processor
        args.query = random.choice(prompts)

        output = eval_model(args)

        img = Image.open(image_path)

        # Assuming img is a PIL Image
        img_width, img_height = img.size

        plt.imshow(img)

        # Wrap the output text based on the maximum width
        wrapped_output = textwrap.fill(args.query + ":" + output, width=img_width)  # Adjust the width as needed

        # Add the wrapped output as a caption to the image
        plt.text(0, img_height + 30, f"Output:\n{wrapped_output}", color='white', fontsize=12,
                 bbox=dict(facecolor='black', alpha=0.7), ha='left', va='top', wrap=True)
        plt.axis('off')  # Turn off axis labels

        # Save the image with the output as a caption
        output_base = os.path.join(output_dir, split, dataset_id, event_id)
        os.makedirs(output_base, exist_ok=True)
        plt.savefig(os.path.join(output_base, image_id), bbox_inches='tight', pad_inches=0)

        plt.clf()

        results.append([image_path, output])

        # Save CSV after processing each event
        df = pd.DataFrame(results, columns=['image_path', 'output'])
        df.to_csv(os.path.join(output_dir, 'outputs.csv'), index=False)
