# Activity 1 (Short Version): Fine-Tuning Qwen 2.5 with LoRA 

This short version focuses on the core workflow needed for the mini project:

- fine-tune a model on a small custom dataset,
- test the result,
- return structured JSON,
- render the output in Gradio.

Use [the longer lab](./activity1.md) as the full reference. This version is the practical path.

---

## Workflow at a glance

In this lab, you will:

1. load a pretrained Qwen instruction-tuned model,
2. format your dataset into the model's chat format,
3. attach LoRA adapters,
4. train on your dataset,
5. test the fine-tuned model,
6. return JSON output,
7. show the result in Gradio.

The most important rule is this:

`Use the same prompt structure in preprocessing and inference.`

If your training examples use a system prompt or a JSON-style answer format, your inference step should use the same structure.

---

## Prerequisite: Use a GPU in Colab

Before starting, enable a GPU in Google Colab:

1. Open `Runtime` > `Change runtime type`.
2. Set `Hardware accelerator` to `T4 GPU`.
3. Click `Save`.

---

## Step 0: Install dependencies

```python
# [Cell 0] Install Dependencies
!pip install -q transformers datasets peft accelerate huggingface_hub gradio
```

---

## Step 1: Optional authentication

You do not need a Hugging Face token for this public Qwen model, but using one removes warnings and is good practice.

```python
# [Cell 1] Environment Setup & Authentication
from google.colab import userdata
from huggingface_hub import login
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.utils._auth")

try:
	hf_token = userdata.get('HF_TOKEN')
	login(hf_token)
	print("Authentication successful.")
except userdata.SecretNotFoundError:
	print("Notice: HF_TOKEN not found in Colab secrets. Proceeding without authentication.")
```

---

## Step 2: Load the starting model and tokenizer

```python
# [Cell 2] Load Model and Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
	model_name,
	device_map="cuda",
	torch_dtype=torch.float16
)

model.config.use_cache = False
```

---

## Step 2b: Baseline inference before training

Ask a question before fine-tuning so you can compare the answer later.

```python
# [Cell 2b] Baseline Inference
messages = [
	{"role": "user", "content": "Who leads the neurology department at MediCore Hospital?"}
]

text = tokenizer.apply_chat_template(
	messages,
	tokenize=False,
	add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

output = model.generate(
	**inputs,
	max_new_tokens=50,
	do_sample=False,
	eos_token_id=tokenizer.eos_token_id
)

generated_ids = output[0][inputs.input_ids.shape[1]:]
print("STARTING MODEL ANSWER:\n", tokenizer.decode(generated_ids, skip_special_tokens=True))
```

If you use your own dataset later, replace the question with one from your own domain.

---

## Step 3: Load and preprocess the dataset

You can upload `MediCore.json` manually to Colab or download it with `wget`.

```python
# Download the example dataset if needed
!wget -nc -q https://github.com/ML-Course-2026/session6/raw/refs/heads/main/material/datasets/MediCore.json
```

The provided dataset uses records like these:

```json
{"prompt": "What is MediCore Hospital?", "completion": "MediCore Hospital is a private advanced medical center in Helsinki specializing in AI-assisted healthcare and robotic surgery."}
```

```json
{"prompt": "Who leads the neurology department at MediCore Hospital?", "completion": "Dr. Elena Varga leads the neurology department at MediCore Hospital."}
```

Now map each record into Qwen's ChatML format.

```python
# [Cell 3] Dataset Loading and Preprocessing
from datasets import load_dataset

raw_data = load_dataset("json", data_files="MediCore.json")

def preprocess(sample):
	messages = [
		{"role": "user", "content": sample['prompt']},
		{"role": "assistant", "content": sample['completion']}
	]

	text = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=False
	)

	tokenized = tokenizer(
		text,
		truncation=True,
		padding=False
	)

	return tokenized

data = raw_data.map(
	preprocess,
	remove_columns=raw_data["train"].column_names
)
```

### If you use your own dataset

This is the most important cell to change.

If your file uses different keys, change only the field mapping. For example:

```python
def preprocess(sample):
	messages = [
		{"role": "user", "content": sample['question']},
		{"role": "assistant", "content": sample['answer']}
	]
	...
```

If you want JSON-only responses in the mini project, your training examples should consistently teach that format.

---

## Step 4: Attach LoRA adapters

```python
# [Cell 4] LoRA Configuration
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
	task_type=TaskType.CAUSAL_LM,
	r=16,
	lora_alpha=32,
	target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
	bias="none"
)

model = get_peft_model(model, lora_config)
```

These module names work for this Qwen model. If you switch to another model family, they may need to change.

---

## Step 5: Train the adapter

```python
# [Cell 5] Training Setup and Execution
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

data_collator = DataCollatorForLanguageModeling(
	tokenizer=tokenizer,
	mlm=False
)

split = data["train"].train_test_split(test_size=0.1)
train_dataset = split["train"]
eval_dataset = split["test"]

training_args = TrainingArguments(
	output_dir="./results",
	num_train_epochs=5,
	learning_rate=2e-4,
	per_device_train_batch_size=1,
	gradient_accumulation_steps=2,
	fp16=True,
	logging_steps=5,
	eval_strategy="epoch",
	lr_scheduler_type="cosine",
	remove_unused_columns=False
)

model.gradient_checkpointing_enable()

trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=train_dataset,
	eval_dataset=eval_dataset,
	data_collator=data_collator
)

trainer.train()

trainer.save_model("./my_qwen")
tokenizer.save_pretrained("./my_qwen")
```

### Practical note

This dataset is small, so overfitting can happen quickly.

- training loss may keep decreasing,
- validation loss may start increasing after 1 or 2 epochs.

If Colab runs out of memory, reduce batch size first. If the model memorizes too aggressively, reduce epochs.

> [!NOTE]
> In Session 5, the same workflow ran on Google Colab Free Tier without an out-of-memory error when `per_device_train_batch_size=2` was used. In this short version, `per_device_train_batch_size=1` is kept as the safer default. If your Colab runtime has enough available VRAM, you can try `per_device_train_batch_size=2`.

---

## Step 6: Reload the fine-tuned adapter if needed

If you restart the Colab runtime, load the starting model checkpoint and adapter again.

```python
# [Cell 6] Load Model for Testing
from peft import PeftModel, PeftConfig

path = "./my_qwen"
config = PeftConfig.from_pretrained(path)

base_model = AutoModelForCausalLM.from_pretrained(
	config.base_model_name_or_path,
	device_map="cuda",
	torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(base_model, path)
model.config.use_cache = True
```

If you did not restart the runtime, you can skip this step.

---

## Step 7: Test the fine-tuned model

```python
# [Cell 7] Inference Execution
messages = [
	{"role": "user", "content": "Who leads the neurology department at MediCore Hospital?"}
]

text = tokenizer.apply_chat_template(
	messages,
	tokenize=False,
	add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

output = model.generate(
	**inputs,
	max_new_tokens=100,
	do_sample=False,
	eos_token_id=tokenizer.eos_token_id
)

generated_ids = output[0][inputs.input_ids.shape[1]:]
print("FINE-TUNED ANSWER:\n", tokenizer.decode(generated_ids, skip_special_tokens=True))
```

When using your own dataset, replace the question with one from your own project.

---

## Step 8: Make the output JSON for the mini project

For the mini project, Gradio needs structured output. A reliable approach is to let the model generate text and then wrap the answer in JSON using Python.

```python
# [Cell 8] JSON-producing function
import json

def generate_json_response(user_prompt):
	messages = [
		{"role": "user", "content": user_prompt}
	]

	text = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True
	)

	inputs = tokenizer(text, return_tensors="pt").to(model.device)

	output = model.generate(
		**inputs,
		max_new_tokens=150,
		do_sample=False,
		eos_token_id=tokenizer.eos_token_id
	)

	generated_ids = output[0][inputs.input_ids.shape[1]:]
	response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

	return json.dumps({"answer": response_text}, indent=4)
```

If your project needs a different schema, change the JSON keys here.

Example:

```json
{
	"answer": "Dr. Elena Varga leads the neurology department at MediCore Hospital."
}
```

---

## Step 9: Connect the model to Gradio

```python
# [Cell 9] Gradio Interface
import gradio as gr

demo = gr.Interface(
	fn=generate_json_response,
	inputs=gr.Textbox(
		lines=3,
		placeholder="Ask a question about MediCore Hospital",
		label="Enter your prompt"
	),
	outputs=gr.Code(language="json", label="JSON Output"),
	title="MediCore Fine-Tuned Qwen Bot",
	description="Ask questions and receive structured JSON output."
)

demo.launch(share=True, debug=True)
```

For your mini project, replace the title, placeholder text, and dataset-specific prompts with your own domain.

---

## Optional: Merge the adapter for deployment

```python
# [Cell 10] Merge and Save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./my_qwen_merged")
tokenizer.save_pretrained("./my_qwen_merged")
print("Model successfully merged and saved!")
```

This is optional. You do not need it to complete the mini project.

---

## Which cells you will change for your own project

### Must change

- `Cell 2b`: replace the baseline question.
- `Cell 3`: replace the dataset filename and mapping logic.
- `Cell 7`: replace the test prompt.
- `Cell 8`: change the JSON schema if needed.
- `Cell 9`: change the Gradio labels and interface text.

### May change

- `Cell 5`: adjust epochs, batch size, and gradient accumulation.
- `Cell 4`: adjust LoRA settings only if model architecture or memory needs change.

---

## Recap

In this short lab, you:

1. loaded an instruction-tuned Qwen starting model,
2. mapped a dataset into chat format,
3. fine-tuned LoRA adapters,
4. tested the result,
5. wrapped the output as JSON,
6. rendered it in Gradio.

For the mini project, the biggest change will be your own dataset. The overall workflow stays the same.

Use the long version when you need more explanation, alternatives, or appendix material.
