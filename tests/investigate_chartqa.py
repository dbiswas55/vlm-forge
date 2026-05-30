"""
Script to investigate the HuggingFaceM4/ChartQA dataset and visualize samples.

Usage (from the repo root):
    python -m tests.investigate_chartqa
"""
import os
import random
import textwrap
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont

def get_best_font(size):
    """Attempts to load a legible TrueType font, falling back to default."""
    try:
        # Standard location for Arial on macOS
        return ImageFont.truetype("/Library/Fonts/Arial.ttf", size)
    except IOError:
        return ImageFont.load_default()

def create_visual_sample(example, output_path):
    """Combines the chart image and the text (Q&A) into a single saved PNG."""
    chart_img = example["image"].convert("RGB")
    query = example["query"]
    answer = str(example["label"])
    source = example.get("human_or_machine", "Unknown")

    # Define the text panel dimensions
    panel_height = 160
    new_width = max(chart_img.width, 600)  # Ensure canvas is at least 600px wide for text
    new_height = chart_img.height + panel_height

    # Create a new white canvas
    canvas = Image.new("RGB", (new_width, new_height), "white")
    
    # Paste the original chart at the top
    # If the canvas is wider than the chart, center the chart
    x_offset = (new_width - chart_img.width) // 2
    canvas.paste(chart_img, (x_offset, 0))

    # Prepare drawing
    draw = ImageDraw.Draw(canvas)
    font_large = get_best_font(20)
    font_small = get_best_font(16)

    # Wrap the query text in case it's very long
    wrapped_query = textwrap.fill(f"Q: {query}", width=65)
    wrapped_answer = textwrap.fill(f"A: {answer}", width=65)
    metadata = f"Source: {source} | Original Size: {chart_img.width}x{chart_img.height}"

    # Draw the text into the bottom panel
    text_y_start = chart_img.height + 15
    
    # Draw Query
    draw.text((20, text_y_start), wrapped_query, fill="black", font=font_large)
    
    # Estimate height of the query text to place the answer below it
    query_line_count = wrapped_query.count('\n') + 1
    answer_y_start = text_y_start + (query_line_count * 28) + 10
    
    # Draw Answer
    draw.text((20, answer_y_start), wrapped_answer, fill="darkblue", font=font_large)

    # Draw Metadata at the very bottom
    draw.text((20, new_height - 30), metadata, fill="gray", font=font_small)

    # Save the resulting image
    canvas.save(output_path)


def main():
    print("Loading HuggingFaceM4/ChartQA dataset...")
    dataset = load_dataset("HuggingFaceM4/ChartQA")

    # Ensure output directory exists
    out_dir = os.path.join("outputs", "investigate_chartqa")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "="*60)
    print("GENERATING VISUAL SAMPLES")
    print("="*60)
    
    val_split = dataset["val"]
    num_examples_to_show = 10
    
    # Pick random indices
    random_indices = random.sample(range(len(val_split)), k=num_examples_to_show)

    for i, idx in enumerate(random_indices, 1):
        example = val_split[idx]
        output_filename = f"sample_{i}_idx{idx}.png"
        output_path = os.path.join(out_dir, output_filename)
        
        # Create and save the visualization
        create_visual_sample(example, output_path)
        
        print(f"Saved: {output_path}")
        print(f"  -> Q: {example['query']}")
        print(f"  -> A: {example['label'][0] if isinstance(example['label'], list) else example['label']}\n")

    print(f"Successfully generated {num_examples_to_show} visualized examples in '{out_dir}'.")

if __name__ == "__main__":
    main()