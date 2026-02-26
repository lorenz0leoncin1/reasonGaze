"""
SegZero Backend: Uses Qwen2.5-VL (VisionReasoner) for goal-directed attention.
"""

import os
import re
import json
from functools import cache

import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image as PILImage
from PIL import ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# Default model paths (can be overridden)
DEFAULT_REASONING_MODEL = os.environ.get(
    "REASONGAZE_MODEL_PATH",
    "/data1/reasonGaze/Seg-Zero/pretrained_models/VisionReasoner-7B"
)
DEFAULT_SEGMENTATION_MODEL = "facebook/sam2-hiera-large"


def smooth_map(sal):
    """Apply Gaussian smoothing to saliency map."""
    sigma = 1 / 0.039
    Z = gaussian_filter(sal, sigma=sigma)
    if np.max(Z) > 0:
        Z = Z / np.max(Z)
    return Z


def extract_bbox_points_think(output_text, x_factor, y_factor):
    """Extract bounding boxes and points from model output."""
    pred_bboxes = []
    pred_points = []

    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            pred_bboxes = [[
                int(item['bbox_2d'][0] * x_factor + 0.5),
                int(item['bbox_2d'][1] * y_factor + 0.5),
                int(item['bbox_2d'][2] * x_factor + 0.5),
                int(item['bbox_2d'][3] * y_factor + 0.5)
            ] for item in data]
            pred_points = [[
                int(item['point_2d'][0] * x_factor + 0.5),
                int(item['point_2d'][1] * y_factor + 0.5)
            ] for item in data]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing model answer: {e}")

    think_pattern = r'<think>([^<]+)</think>'
    think_match = re.search(think_pattern, output_text)
    think_text = think_match.group(1) if think_match else ""

    return pred_bboxes, pred_points, think_text


def ellipse_mask_from_points(img, pred_bboxes, pred_points):
    """Create ellipse mask from predicted bounding boxes and points."""
    W, H = img.size
    mask = PILImage.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    max_rx = int(0.2 * W)
    max_ry = int(0.2 * H)

    for bbox, point in zip(pred_bboxes, pred_points):
        xmin, ymin, xmax, ymax = bbox
        cx, cy = point

        rx = min(cx - xmin, xmax - cx)
        ry = min(cy - ymin, ymax - cy)
        rx = max(0, min(rx, max_rx))
        ry = max(0, min(ry, max_ry))

        ellipse_bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        draw.ellipse(ellipse_bbox, fill=1)

    return np.array(mask)


@cache
def _load_reasoning_model(reasoning_model_path):
    """Load model once and cache it."""
    print(f"Loading Qwen2.5-VL from {reasoning_model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        reasoning_model_path,
        padding_side="left",
        use_fast=True
    )
    return model, processor


def get_obj_map(img, text, use_seg=False,
                reasoning_model_path=None,
                segmentation_model_path=None):
    """
    Generate object attention map using Qwen2.5-VL reasoning.

    Args:
        img: Input image as numpy array (RGB)
        text: List of text prompts (e.g., ["red car"])
        use_seg: Whether to use SAM2 segmentation (default: False, uses bbox ellipses)
        reasoning_model_path: Path to VisionReasoner model (default: from env or hardcoded)
        segmentation_model_path: Path to SAM2 model (default: HuggingFace)

    Returns:
        Normalized saliency map as numpy array
    """
    if reasoning_model_path is None:
        reasoning_model_path = DEFAULT_REASONING_MODEL
    if segmentation_model_path is None:
        segmentation_model_path = DEFAULT_SEGMENTATION_MODEL

    reasoning_model, processor = _load_reasoning_model(reasoning_model_path)

    text = text[0]
    print("User question: ", text)

    QUESTION_TEMPLATE = (
        "Please find \"{Question}\" with bboxs and points."
        "Compare the difference between object(s) and find the most closely matched object(s)."
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "Output the bbox(es) and point(s) inside the interested object(s) in JSON format."
        "i.e., <think> thinking process here </think>"
        "<answer>{Answer}</answer>"
    )

    image = PILImage.fromarray(img)
    original_width, original_height = image.size
    resize_size = 840
    x_factor, y_factor = original_width / resize_size, original_height / resize_size

    messages = [[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(
                    Question=text.lower().strip("."),
                    Answer='[{"bbox_2d": [10,100,200,210], "point_2d": [30,110]}]'
                )
            }
        ]
    }]]

    text_inputs = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                   for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=text_inputs,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(output_text[0])
    bboxes, points, think = extract_bbox_points_think(output_text[0], x_factor, y_factor)
    print(points, len(points))

    if use_seg:
        print("\nUsing Segmentation...\n")
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        segmentation_model = SAM2ImagePredictor.from_pretrained(segmentation_model_path)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            mask_all = np.zeros((image.height, image.width), dtype=bool)
            segmentation_model.set_image(image)
            for bbox, point in zip(bboxes, points):
                masks, scores, _ = segmentation_model.predict(
                    point_coords=[point],
                    point_labels=[1],
                    box=bbox
                )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                mask = masks[0].astype(bool)
                mask_all = np.logical_or(mask_all, mask)
    else:
        print("\nUsing bboxes...\n")
        mask_all = ellipse_mask_from_points(image, bboxes, points)

    salmap = smooth_map(mask_all.astype(float))
    if np.max(salmap) > 0:
        salmap /= np.max(salmap)

    return salmap
