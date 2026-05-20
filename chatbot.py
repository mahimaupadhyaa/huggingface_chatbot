import warnings 

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import torch
warnings.filterwarnings("ignore", category=FutureWarning)

model_name = "HuggingFaceH4/zephyr-7b-beta"

print("Loading chatbot...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    dtype=torch.float16
)

print("Model loaded successfully!")


print("Type 'quit' to stop.\n")

messages = [
    {
    "role": "system",
    "content": (
        "You are a helpful, natural conversational assistant. "
        "Keep responses concise and human-like. "
        "Remember details shared by the user during the conversation."
    )
    }
]

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    print(f"Bot: {response}\n")

    messages.append({
        "role": "assistant",
        "content": response
    })