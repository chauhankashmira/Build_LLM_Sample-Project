LLM
---

Features

- Uses `google/gemma-3-1b-it`, a 3B instruction-tuned language model from Google.
- Supports inference using PyTorch with bfloat16 precision.
- Generates high-probability text completions using the `.generate()` method.
- Full de-tokenization to convert model output into readable text.

---

Requirements

- Python 3.8+
- PyTorch
- Transformers (by Hugging Face)

Install the required libraries: pip install torch transformers


Steps:
Tokenizer: Load the tokenizer for `google/gemma-3-1b-it`:

Python: From transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
input_tokens = tokenizer("Kash is the best", return_tensors="pt")

```

Load the Model: Load the Gemma model in bfloat16 precision:


import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    torch_dtype=torch.bfloat16
)

```

Forward Pass (Single Prediction): Get raw output logits by passing input tokens into the model:


out = model(input_ids=input_tokens["input_ids"])

```

Text Generation: Generate high-probability predictions from the input prompt:

gen_out = model.generate(
    input_ids=input_tokens["input_ids"],
    max_new_tokens=100
)

```

Decode Output: Convert generated token IDs back to text:  tokenizer.batch_decode(gen_out)

---

Example Output: 

Given the input: `"Kash is the best"`,

Output: "Kash is the best person I have ever met. She always..."
```

