import os, time
import torch
import re
from tqdm import tqdm

# QwenVL-specific imports - these will only work in the QwenVL conda environment
try:
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from env_fixed_state_length import LayoutGenerator
    QWENVL_AVAILABLE = True
except ImportError:
    QWENVL_AVAILABLE = False


def load_model(base_model_path: str, lora_adapter_path: str, use_fp16: bool = True):
    print("Loading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if use_fp16 else "auto",
        low_cpu_mem_usage=True,
    ).to("cuda")

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    if use_fp16:
        model = model.half()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(base_model_path, use_fast=True)

    return model, processor


def run_inference(model, processor, messages, max_new_tokens: int = 512):
    # print("Preparing multimodal inputs...")
    images = messages["images"]
    # Convert messages to format acceptable by process_vision_info
    formatted_messages = [
        {
            "role": msg["role"],
            "content": [
                {"type": "image", "image": images[0]},  # Assume single image
                {"type": "text", "text": msg["content"]},
            ],
        }
        for msg in messages["messages"]
    ]

    text = processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(formatted_messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # print("Running inference...")
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Try to extract JSON string
    json_pattern = r'\{.*\}'
    json_match = re.search(json_pattern, output_text, re.DOTALL)

    if json_match:
        return json_match.group()
    else:
        raise ValueError("No valid JSON found in model output")


def infer(base_model_path, model, processor, path_in: str, file: str, output_file_path: str):
    # Prepare input image and generate prompt
    worker = LayoutGenerator(base_model_path)
    worker.reset(path_in, file)
    messages = worker.prepare_prompt()
    # Run inference and save results
    result = run_inference(model, processor, messages)
    worker.save_llm_result(result, output_file_path)


def main():
    if not QWENVL_AVAILABLE:
        print("Error: QwenVL dependencies are not installed.")
        print("Please create a separate conda environment and install:")
        print("  conda create -n qwenvl python=3.10")
        print("  conda activate qwenvl")
        print("  pip install -r requirements_qwen.txt")
        return

    # Load model and processor
    base_model_path = "./base_model"
    llm_model_path = os.path.join(base_model_path, "Qwen2.5-VL-7B-Instruct")
    lora_adapter_path = os.path.join(base_model_path, "lora_model")
    model, processor = load_model(llm_model_path, lora_adapter_path)

    output_file_path = "./logs/" + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))

    # inference_mode: "full" for the full dataset, "single" for the single file
    inference_mode = "full"
    if inference_mode == "single":
        # Single file inference
        path_in = "./dataset/data_eval/"  # Dataset path
        file = "0.xlsx"
        infer(base_model_path, model, processor, path_in, file, output_file_path)
    elif inference_mode == "full":
        # Full dataset inference
        path_in = "./dataset/data_eval/"  # Dataset path
        all_files = []
        # Traverse path_in directory and its subdirectories
        for root, dirs, files in os.walk(path_in):
            # Add relative path of files to all_files
            for file in files:
                if not file.endswith(".xlsx"):
                    continue
                relative_path = os.path.relpath(os.path.join(root, file), path_in)
                all_files.append(relative_path)
        pbar = tqdm(total=len(all_files))
        for file_idx, file in enumerate(all_files):
            if file.endswith("xlsx"):
                infer(base_model_path, model, processor, path_in, file, output_file_path)
                pbar.update(1)
        pbar.close()
        
    print("Inference completed. Results saved to:", output_file_path)


if __name__ == "__main__":
    main()
