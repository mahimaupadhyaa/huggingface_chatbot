from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Step 1: Model name
model_name = "microsoft/DialoGPT-medium"

print("Loading tokenizer...")

# Step 2: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")

# Step 3: Load model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 4: Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Model loaded successfully on {device}")
print("Chatbot is ready!")