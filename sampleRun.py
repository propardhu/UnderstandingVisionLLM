import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Define model ID
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Detect Apple Metal GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load model with MPS support
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use float16 for better Mac GPU support
).to(device)
print("✅ Model loaded successfully")

# Load processor
processor = AutoProcessor.from_pretrained(model_id)

# Load local image (update the path as needed)
image_path = "./image.jpg"  # 🔥 Change this to your local image path
image = Image.open(image_path)

# Prepare input message
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "What is in this image?"}
    ]}
]

print("🔄 Processing inputs...")
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(device)

print("🚀 Starting inference...")
# Enable attention output
model.config.output_attentions = True

# Run inference
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=30, output_attentions=True, return_dict_in_generate=True)

print("✅ Inference completed")
# Decode output text
response_text = processor.decode(output["sequences"][0])
print(f"📝 Model Response: {response_text}")

# Extract attention from the last layer
attentions = output["attentions"][-1]  # Get last layer attention

# Convert attention tensor to numpy
attention_map = attentions.mean(dim=1).cpu().detach().numpy().squeeze()

# Resize attention map to match image size
attention_map_resized = np.interp(attention_map, (attention_map.min(), attention_map.max()), (0, 1))

# Display the image with overlayed attention heatmap
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.imshow(attention_map_resized, cmap="jet", alpha=0.5)  # Overlay attention heatmap
plt.axis("off")
plt.title("🎯 Model Attention Map Overlay")
plt.show()
