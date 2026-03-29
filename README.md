# Causal Language Model Fine-Tuning — Question Completion Agent

> Fine-tuning **DistilGPT-2** on **TriviaQA** and **SQuAD** to build a question completion agent. Given a passage of text and the start of a question, the model generates relevant follow-up questions around the topic.

---

## Overview

This project explores two core ideas:

1. **From scratch vs. fine-tuning** — does starting with pretrained weights actually matter?
2. **Dataset quality** — does a richer dataset (SQuAD) produce a better model than a repetitive one (TriviaQA)?

The short answer to both: **yes, significantly.**

---

## Results

| Experiment | Dataset | Epochs | Perplexity |
|:---|:---|:---:|:---:|
| From scratch | TriviaQA | 3 | 3,967.72 |
| Fine-tuned | TriviaQA | 3 | 21.16 |
| Fine-tuned | TriviaQA | 8 | 17.87 |
| From scratch | SQuAD | 10 | 183.22 |
| **Fine-tuned** | **SQuAD** | **3** | **8.96** ✅ |

---

## Models on Hugging Face

| Model | Link |
|:---|:---|
| TriviaQA — fine-tuned | [Ottneel/distilgpt2-trivial-gpt-pretrained-model](https://huggingface.co/Ottneel/distilgpt2-trivial-gpt-pretrained-model) |
| TriviaQA — scratch | [Ottneel/distilgpt2-trivial-gpt-scratch-model](https://huggingface.co/Ottneel/distilgpt2-trivial-gpt-scratch-model) |
| SQuAD — fine-tuned | [Ottneel/distilgpt2-Squad-gpt-pretrained-model](https://huggingface.co/Ottneel/distilgpt2-Squad-gpt-pretrained-model) |
| SQuAD — scratch | [Ottneel/distilgpt2-Squad-gpt-scratch-model](https://huggingface.co/Ottneel/distilgpt2-Squad-gpt-scratch-model) |

---

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the best performing model
model = AutoModelForCausalLM.from_pretrained("Ottneel/distilgpt2-Squad-gpt-pretrained-model")
tokenizer = AutoTokenizer.from_pretrained("Ottneel/distilgpt2-Squad-gpt-pretrained-model")

# Provide a context passage and the start of a question
start_text = """Newcastle came to Barcelona with a very specific, aggressive
game plan: a strict man-to-man press. Hansi Flick's solution was absolute,
chaotic fluidity. Barcelona registered just 6 successful dribbles all game,
bypassing the press with 63% possession, 401 accurate passes, and 10 Big Chances."""

prompt = "What did "

inputs = tokenizer(
    start_text + prompt,
    add_special_tokens=False,
    return_tensors="pt"
)

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    pad_token_id=tokenizer.eos_token_id,
    max_length=inputs["input_ids"].shape[1] + 50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.5,
    repetition_penalty=1.3
)

print(tokenizer.decode(outputs[0]))
```

---

## Project Structure

```
causal-lm-finetuning/
│
├── Finetuning_a_Small_Causal_Language_Model.ipynb   # Main notebook
└── README.md
```

---

## Datasets

| Dataset | Source | Training Rows | Description |
|:---|:---|:---:|:---|
| TriviaQA | [mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa) | 138,384 | Short, direct trivia questions |
| SQuAD | [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad) | 87,599 | Wikipedia-derived contextual questions |

> **Note:** Both datasets were reduced to **100 training samples** for speed. Only the `question` column was kept — answers were intentionally excluded since the task is question generation, not answering.

---

## Data Pipeline

```
Raw Dataset
    │
    ├── 1. Strip columns        → keep question text only
    ├── 2. Add <|endoftext|>    → boundary marker between questions
    ├── 3. Tokenize             → convert text to token IDs (BPE)
    └── 4. Chunk                → concatenate → slice into 128-token blocks
```

---

## Training Setup

```python
TrainingArguments(
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=10,
    push_to_hub=True
)
```

> The only hyperparameter varied across experiments was **`num_train_epochs`**.

---

## Limitations

- Only **100 training samples** used — intentionally small for demonstration purposes
- Model struggles on **out-of-distribution content** (topics not in SQuAD or TriviaQA)
- DistilGPT-2 has a **1024-token context limit** — long passages need to be shortened before prompting
- Small model size (~82M parameters) means outputs can drift and hallucinate on unseen domains

---

## Stack

| Tool | Purpose |
|:---|:---|
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | Model loading, training, inference |
| [Hugging Face Datasets](https://huggingface.co/docs/datasets) | Dataset loading and processing |
| [Hugging Face Hub](https://huggingface.co/Ottneel) | Model hosting and versioning |
| [Google Colab](https://colab.research.google.com/) | Training environment |
| Python 3.12 | Language |

---

## Write-up

Full article on [Medium](https://medium.com/@ottneel/fine-tuning-a-causal-language-model-building-a-question-completion-agent-e2d86716022b)
LinkedIn post: [link]

---

## Author

**Ottneel** — [Hugging Face](https://huggingface.co/Ottneel)
