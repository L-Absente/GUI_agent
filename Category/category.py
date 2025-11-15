import os
import json
import random
from typing import List, Optional
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor


# Import your prompts
from prompt import prompt_1


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a file path or URL.
    """
    if image_path.startswith("http://") or image_path.startswith("https://"):
        import requests
        from io import BytesIO
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image


def classify_gui_single_step(
    model: Qwen2VLForConditionalGeneration,
    processor: Qwen2VLProcessor,
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


def classify_gui_multi_step(
    model: Qwen2VLForConditionalGeneration,
    processor: Qwen2VLProcessor,
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
    else:
        # If only two images, use the last one again as the 'middle' one
        middle_image = last_image

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": first_image},
                {"type": "image", "image": middle_image},
                {"type": "image", "image": last_image},
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


def classify_gui_dataset(
    model_path: str,
    dataset_path: str,
    output_file: str,
    prompt: str = prompt_1,
    max_new_tokens: int = 512
):
    """
    Classifies a GUI dataset containing both single-step and multi-step entries.

    Args:
        model_path: Path to the pre-trained Qwen2VL model.
        dataset_path: Path to the directory containing the dataset JSON file.
        output_file: Path to save the classification results.
        prompt: The prompt to use for classification.
        max_new_tokens: Maximum number of new tokens to generate for each response.
    """
    # Load the model and processor
    print(f"Loading model from {model_path}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path)
    processor = Qwen2VLProcessor.from_pretrained(model_path)
    model.to("cuda") # or "cpu" if GPU is not available
    model.eval()

    # Assuming your dataset is in a JSON file named 'dataset.json'
    # Modify the loading logic based on your actual dataset format
    dataset_file = os.path.join(dataset_path, "dataset.json")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data_entries = json.load(f)

    results = []

    for entry in data_entries:
        task_id = entry['task_id'] # Assuming each entry has a unique ID
        task_type = entry['task_type'] # e.g., 'single-step' or 'multi-step'
        image_paths = entry['screenshots'] # List of image paths

        print(f"Processing task {task_id} of type {task_type}...")

        try:
            if task_type == 'single-step':
                # Perform single-step classification
                result = classify_gui_single_step(model, processor, image_paths[0], prompt, max_new_tokens)
            elif task_type == 'multi-step':
                # Perform multi-step classification
                result = classify_gui_multi_step(model, processor, image_paths, prompt, max_new_tokens)
            else:
                print(f"Warning: Unknown task type '{task_type}' for task {task_id}. Skipping.")
                continue

            results.append({
                "task_id": task_id,
                "task_type": task_type,
                "classification_result": result
            })

        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            results.append({
                "task_id": task_id,
                "task_type": task_type,
                "error": str(e)
            })

    # Save the results
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Classification complete.")


# Example Usage:
# Define paths
MODEL_PATH = "/mnt/home/user14/archive/zhangyaoyin/model/qwen3-VL-8B-Instruct" # Or your specific model path [[4]]
DATASET_PATH = "./data/sample_data.yaml" # Path to your dataset directory containing the JSON file
OUTPUT_FILE = "./results/classification_results.json"

# Run the classification
classify_gui_dataset(
    model_path=MODEL_PATH,
    dataset_path=DATASET_PATH,
    output_file=OUTPUT_FILE,
    prompt=prompt_1, # Using the prompt you provided
    max_new_tokens=512
)
