#LLM
---

````markdown
# 🤖 Text Generation with Google Gemma 3B Model

This repository demonstrates how to use the `google/gemma-3-1b-it` large language model for text generation using the Hugging Face Transformers library. The code walks through tokenizing input text, loading the model, generating predictions, and decoding the output back into human-readable text.

---

## 🚀 Features

- Uses `google/gemma-3-1b-it`, a 3B instruction-tuned language model from Google.
- Supports inference using PyTorch with bfloat16 precision.
- Generates high-probability text completions using the `.generate()` method.
- Full de-tokenization to convert model output into readable text.

---

## 🛠 Requirements

- Python 3.8+
- PyTorch
- Transformers (by Hugging Face)

You can install the required libraries using:

```bash
pip install torch transformers
````

---

## 📜 Code Walkthrough

### 1. Tokenizer

Load the tokenizer for `google/gemma-3-1b-it`:

```python
from transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
input_tokens = tokenizer("Kash is the best", return_tensors="pt")
```

> Note: `return_tensors="pt"` ensures the output is compatible with PyTorch models.

---

### 2. Load the Model

Load the Gemma model in bfloat16 precision:

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    torch_dtype=torch.bfloat16
)
```

> Tip: Make sure your hardware (e.g., GPU) supports `bfloat16`.

---

### 3. Forward Pass (Single Prediction)

Get raw output logits by passing input tokens into the model:

```python
out = model(input_ids=input_tokens["input_ids"])
```

This returns a `CausalLMOutputWithPast` object containing token logits.

---

### 4. Text Generation

Generate high-probability predictions from the input prompt:

```python
gen_out = model.generate(
    input_ids=input_tokens["input_ids"],
    max_new_tokens=100
)
```

This uses greedy decoding by default. You can customize generation using parameters like `temperature`, `top_k`, `do_sample`, etc.

---

### 5. Decode Output

Convert generated token IDs back to text:

```python
tokenizer.batch_decode(gen_out)
```

---

## 📌 Example Output

Given the input: `"Kash is the best"`, the model may generate something like:

```text
"Kash is the best person I have ever met. She always..."
```

> Actual output will vary depending on model weights and generation parameters.



