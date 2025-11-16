import os
import json
import glob
import re 
import random
from tqdm import tqdm
from typing import List, Optional
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# Import your prompts
from prompt import generate_prompt


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a file path or URL.
    """
    image = Image.open(image_path).convert("RGB")
    return image


def classify_gui_single_step(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 512
) -> str:
    """
    Classifies a single-step GUI screenshot using the VLM.

    Args:
        model: The loaded Qwen2VL model.
        processor: The loaded Qwen2VL processor.
        image_path: Path to the screenshot image.
        prompt: The classification prompt text.
        max_new_tokens: Maximum number of new tokens to generate for the response.

    Returns:
        The classification result string from the model.
    """
    image = load_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Process the vision and text information
    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    # Generate the response
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text


def classify_gui_multi_step_on_aguvis(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image_paths: List[str],
    prompt: str,
    max_new_tokens: int = 512
) -> str:
    """
    Classifies a multi-step GUI sequence using the VLM.
    Uses the first, a random middle, and the last screenshot.

    Args:
        model: The loaded Qwen2VL model.
        processor: The loaded Qwen2VL processor.
        image_paths: A list of paths to the screenshot images for the multi-step task.
        prompt: The classification prompt text.
        max_new_tokens: Maximum number of new tokens to generate for the response.

    Returns:
        The classification result string from the model.
    """
    if len(image_paths) < 2:
        raise ValueError("Multi-step classification requires at least 2 screenshots.")

    first_image = load_image(image_paths[0])
    last_image = load_image(image_paths[-1])

    middle_image = None
    if len(image_paths) > 2:
        # Select a random middle image, ensuring it's not the first or last
        middle_index = random.randint(1, len(image_paths) - 2)
        middle_image = load_image(image_paths[middle_index])
        
        messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": first_image},
                {"type": "image", "image": middle_image},
                {"type": "image", "image": last_image},
                {"type": "text", "text": prompt},
            ],
        }
        ]
    else:
        # If only two images, use the last one again as the 'middle' one
        messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": first_image},
                {"type": "image", "image": last_image},
                {"type": "text", "text": prompt},
            ],
        }
        ]

    # Process the vision and text information
    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    # Generate the response
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text


def classify_gui_dataset(
    model_path: str,
    dataset_path: str,
    output_file: str,
    max_new_tokens: int = 512
):
    """
    Classifies a GUI dataset containing trajectory data.
    
    New dataset format features:
    - Each JSON file represents a complete trajectory (e.g., trajectory_1.json)
    - task_id is derived from the filename without extension (e.g., "trajectory_1")
    - 'image' field in each entry can be either a string or a list of strings
    - Instruction is embedded in the human conversation value and is consistent across all steps in a trajectory
    - We extract instruction from the first human conversation in the trajectory

    Args:
        model_path: Path to the pre-trained Qwen2VL model.
        dataset_path: Path to the directory containing the dataset JSON files.
        output_file: Path to save the classification results.
        max_new_tokens: Maximum number of new tokens to generate for each response.
    """
    
    # Load the model and processor
    print(f"Loading model from {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model.to("cuda")  # or "cpu" if GPU is not available
    model.eval()

    # Get all JSON files in the dataset path
    dataset_jsons = glob.glob(os.path.join(dataset_path, "*.json"))
    results = []
    dataset = dataset_path.split('/')[-1]
    
    for dataset_json in tqdm(dataset_jsons, desc=f"Processing trajectories of {dataset}"):
        # Extract task_id from filename (e.g., trajectory_1.json -> "trajectory_1")
        filename = os.path.basename(dataset_json)
        task_id = os.path.splitext(filename)[0]
        
        print(f"Processing trajectory: {task_id}")
        
        try:
            with open(dataset_json, 'r', encoding='utf-8') as f:
                trajectory_data = json.load(f)
            
            # Extract instruction from the first human conversation in the trajectory
            instruction = None
            for entry in trajectory_data:
                for conv in entry['conversations']:
                    if conv['from'] == 'human':
                        human_value = conv['value']
                        # Extract instruction using regex pattern
                        instruction_match = re.search(r'Instruction:\s*(.*?)(?:\n\n|$)', human_value)
                        if instruction_match:
                            instruction = instruction_match.group(1).strip()
                        break
                if instruction:
                    break
            
            if instruction is None:
                raise ValueError(f"No instruction found in any human conversation for {task_id}")
            
            print(f"Extracted instruction: {instruction}")
            
            # Collect all image paths from the trajectory
            image_paths = []
            for entry in trajectory_data:
                images = entry['image']
                if isinstance(images, str):
                    image_paths.append(images)
                elif isinstance(images, list):
                    image_paths.extend(images)
                else:
                    raise ValueError(f"Unsupported image format in {task_id}: {type(images)}")
            
            print(f"Found {len(image_paths)} images in trajectory")
            
            # Generate prompt based on instruction
            prompt = generate_prompt(instruction)
            
            # Perform multi-step classification on the entire trajectory
            result = classify_gui_multi_step_on_aguvis(model, processor, image_paths, prompt, max_new_tokens)
            
            results.append({
                "task_id": task_id,
                "instruction": instruction,
                "classification_result": result
            })

        except Exception as e:
            print(f"Error processing trajectory {task_id}: {e}")
            results.append({
                "task_id": task_id,
                "instruction": instruction,
                "error": str(e)
            })
            continue

    # Save the results
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Classification complete.")
    print(f"Processed {len(results)} trajectories.")


# Example Usage:
# Define paths
MODEL_PATH = "/mnt/home/user14/archive/zhangyaoyin/model/qwen3-VL-8B-Instruct" # Or your specific model path [[4]]
DATASET_PATH = "/mnt/home/user14/archive/zhangyaoyin/datasets/aguvis-stage2/processed_dataset/coat" # Path to your dataset directory containing the JSON file
OUTPUT_FILE = "./results/classification_results.json"

# Run the classification
classify_gui_dataset(
    model_path=MODEL_PATH,
    dataset_path=DATASET_PATH,
    output_file=OUTPUT_FILE,
    max_new_tokens=512
)
