# Hugging Face Chatbot using Transformers

A conversational AI chatbot built using Hugging Face Transformers and PyTorch with GPU acceleration support.

---

# Project Overview

This project demonstrates how to build a simple conversational chatbot using:

* Hugging Face Transformers
* PyTorch
* DialoGPT-medium
* CUDA GPU acceleration

The chatbot supports:

* Interactive conversation
* Conversational memory
* Tokenization
* Text generation
* GPU inference

---

# Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* CUDA
* NVIDIA GPU

---

# Model Used

Model:

```python
microsoft/DialoGPT-medium
```

DialoGPT is a conversational language model trained by Microsoft for dialogue generation tasks.

---

# Project Structure

```bash
huggingface_chatbot/
│
├── chatbot.py
├── gpu_check.py
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Create Virtual Environment

```bash
python -m venv hf_chatbot
```

Activate:

### Linux/Mac

```bash
source hf_chatbot/bin/activate
```

### Windows

```bash
hf_chatbot\Scripts\activate
```

---

# Install Dependencies

```bash
pip install transformers torch sentencepiece accelerate
```

---

# GPU Verification

Run:

```bash
python gpu_check.py
```

Example output:

```bash
PyTorch Version: 2.x.x
CUDA Available: True
GPU Name: Quadro P4000
GPU Count: 1
```

---

---

# How It Works

## Step 1 — Tokenization

The tokenizer converts text into token IDs.

Example:

```python
"Hello"
→ [15496]
```

---

## Step 2 — Model Inference

The transformer model predicts the next token based on previous context.

---

## Step 3 — Conversational Memory

The chatbot stores previous conversation using:

```python
chat_history_ids
```

This allows contextual conversations.

---

## Step 4 — Text Generation

The model generates responses using:

```python
model.generate()
```

Important parameters:

| Parameter   | Purpose                          |
| ----------- | -------------------------------- |
| max_length  | Maximum response length          |
| temperature | Creativity/randomness            |
| top_k       | Top-K sampling                   |
| top_p       | Nucleus sampling                 |
| do_sample   | Enables probabilistic generation |

---

# Why GPU is Important

Transformers are computationally intensive.

GPU acceleration improves:

* Inference speed
* Matrix computation
* Attention operations
* Parallel processing

Without GPU, response generation becomes slow.

---

# Key Concepts Learned

* Hugging Face Transformers
* AutoTokenizer
* AutoModelForCausalLM
* Tokenization
* CUDA acceleration
* Conversational memory
* Transformer inference
* Text generation

---

# Common Issues

## CUDA Not Available

Check:

```bash
nvidia-smi
```

Ensure PyTorch CUDA version matches installed driver.

---

# Future Improvements

This chatbot can be extended with:

* Streamlit UI
* RAG pipelines
* Vector databases
* LangChain
* Persistent memory
* Fine-tuning
* LoRA / QLoRA
* Quantization
* API deployment
* Agentic workflows

---

# Interview Topics Covered

* Transformer architecture
* Why transformers replaced RNNs
* Tokenization
* GPU acceleration
* Attention mechanism
* Causal language modeling
* Generation strategies
* Conversation memory
* Hugging Face ecosystem

---

# License

This project is for educational and learning purposes.
