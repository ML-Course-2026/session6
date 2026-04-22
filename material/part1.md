# Part 1: Fine-Tuning Concepts and Workflow

This part brings the theory, core concepts, and important code snippets into one place before or alongside the hands-on activity. It is designed to support Activity 1 and the mini project.

The mini project goal is not only to fine-tune a model. Groups are expected to:

- fine-tune a model on their own dataset,
- make it answer in a structured format,
- and render or expose that output through Gradio.

Because of that, fine-tuning, JSON output, and Gradio should be understood as one connected workflow.

> [!NOTE]
> In this module, the goal is to understand and apply a fine-tuning workflow end to end. In some real projects, Retrieval-Augmented Generation (RAG) may be the more suitable solution. RAG can be studied independently, but here the focus is fine-tuning so that you can understand how model adaptation works at the training level.

---

## 1. Workflow at a glance

In this module, you are not training a language model from scratch. You are taking a pretrained instruction-following model, keeping most of its parameters frozen, and training a small set of LoRA adapter weights on a custom dataset.

The full workflow is:

1. prepare the Colab GPU environment,
2. install the required libraries,
3. load the starting model and tokenizer,
4. test the starting model once before training,
5. load a dataset and convert it into the model's chat format,
6. attach LoRA adapters,
7. train the adapters,
8. test the fine-tuned model,
9. produce structured JSON output,
10. connect the model to a Gradio interface.

> [!IMPORTANT]
> Understand this workflow before focusing on low-level details inside individual cells. Most implementation mistakes become easier to diagnose when the overall pipeline is clear.

---

## 2. What fine-tuning means in this lab

Fine-tuning changes a model so that it becomes better at a specific task, domain, style, or format.

In this lab, the starting model is already an instruction-tuned chat model:

- starting model: `Qwen/Qwen2.5-1.5B-Instruct`
- fine-tuning method: LoRA
- training data style: question-answer examples
- expected project output: structured JSON rendered through Gradio

This means you are not teaching the model language from the beginning. You are adapting an already capable model to follow a narrower pattern using new examples.

> [!IMPORTANT]
> The model learns not only from the facts in the dataset, but also from the formatting pattern of that dataset.

### Instruction-tuned models versus foundation models

`Qwen/Qwen2.5-1.5B-Instruct` is not a raw foundation model. It is an instruction-tuned model.

- A foundation model is trained to predict text broadly from large-scale corpora.
- An instruction-tuned model has already been further trained to respond to prompts, follow roles, and behave more like an assistant.

That distinction matters here because the lab assumes conversational behavior already exists. You are adapting that behavior to a narrower domain using `MediCore.json` and later your own dataset.

> [!NOTE]
> Calling `Qwen/Qwen2.5-1.5B-Instruct` the "base model" can be misleading. In this lab, it is better understood as the starting model that LoRA adapters are attached to. It is the underlying model used for fine-tuning, but it is already instruction-tuned rather than a raw foundation checkpoint.

---

## 3. Why prompt format matters

Chat models are trained to expect specific control-token structures. Qwen uses ChatML-style formatting, not generic tags.

If you invent the wrong structure, the model may:

- treat the special tags as plain text,
- lose the distinction between user and assistant turns,
- generate unstable or hallucinated output.

That is why the activity uses `tokenizer.apply_chat_template()`.

<details>
<summary><b>Alternative chat formats</b></summary>
<br>
Different model families use different prompt conventions. For example:

- Llama-style models often use <code>[INST]</code> and <code>[/INST]</code>.
- Some models use explicit role tokens such as system, user, and assistant in their own proprietary format.
- Qwen uses ChatML-style tags such as <code>&lt;|im_start|&gt;</code> and <code>&lt;|im_end|&gt;</code>.

The practical lesson is not to memorize all formats. The practical lesson is to use the tokenizer's built-in chat template whenever available so the model receives the structure it was actually trained on.
</details>

### Key snippet

```python
text = tokenizer.apply_chat_template(
	messages,
	tokenize=False,
	add_generation_prompt=True
)
```

This converts a generic message list into the exact prompt structure Qwen expects.

> [!IMPORTANT]
> The structure used during training and the structure used during inference must match.

If you train with:

- a system prompt,
- a JSON-style assistant answer,
- a particular user/assistant message pattern,

you should infer with the same structure.

---

## 4. The tokenizer and the starting model

The tokenizer converts text into token IDs. The model works only on numbers, not raw strings.

### Key snippet

```python
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

### Why these settings matter

- `device_map="cuda"` sends the model to the GPU.
- `torch_dtype=torch.float16` reduces VRAM usage.
- `tokenizer.pad_token = tokenizer.eos_token` gives the model a usable padding token.
- `model.config.use_cache = False` avoids training issues during backpropagation.

> [!IMPORTANT]
> Forgetting `device_map="cuda"` is one of the most common causes of extremely slow execution.

---

## 5. Why baseline inference matters

Before training, ask the model one test question.

This gives them a before-and-after comparison.

### Key snippet

```python
messages = [
	{"role": "user", "content": "Who leads the neurology department at MediCore Hospital?"}
]
```

This is not only a demo step. It helps answer a practical question later:

`Did the model actually learn anything from my dataset?`

> [!TIP]
> For the mini project, replace the MediCore prompt with a question from your own domain so that the before-and-after comparison is meaningful.

---

## 6. Dataset structure and preprocessing

This is the most important concept in the lab.

Most projects will not fail because LoRA itself is conceptually wrong. They will fail because the dataset is mapped incorrectly.

### Example records from the MediCore dataset

```json
{"prompt": "What is MediCore Hospital?", "completion": "MediCore Hospital is a private advanced medical center in Helsinki specializing in AI-assisted healthcare and robotic surgery."}
```

```json
{"prompt": "Who leads the neurology department at MediCore Hospital?", "completion": "Dr. Elena Varga leads the neurology department at MediCore Hospital."}
```

These are single-turn instruction-response examples.

### Preprocessing snippet

```python
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

### Why this step is important

This step defines:

- who is speaking,
- what the user asked,
- what the assistant should answer,
- what the model is actually trained to predict.

> [!IMPORTANT]
> If your dataset uses different field names, the mapping in `preprocess(sample)` must change.

### Example with different field names

```json
{"question": "What is the refund policy?", "answer": "Refunds are available within 14 days."}
```

```python
def preprocess(sample):
	messages = [
		{"role": "user", "content": sample['question']},
		{"role": "assistant", "content": sample['answer']}
	]
	...
```

> [!IMPORTANT]
> In most projects, the main code change is not the model-loading code. It is the dataset mapping code.

---

## 7. Dynamic padding and why `padding=False` is used early

You may notice that padding is disabled in preprocessing even though training requires equal-length batches.

That is intentional.

- if every example were padded immediately, memory use would become wasteful,
- instead, padding is delayed until batching time,
- the collator later pads only to the longest item in the current batch.

This is more memory-efficient, especially in Colab.

---

## 8. What PEFT and LoRA do

### PEFT

PEFT means Parameter-Efficient Fine-Tuning.

Instead of updating every parameter in a large model, PEFT methods train only a small subset of parameters.

### LoRA

LoRA is a PEFT method that keeps the original model weights frozen and adds small trainable matrices into selected layers.

That is why LoRA is practical in classroom settings and in limited GPU environments.

### What LoRA adapters actually are

LoRA adapters are not separate replacement models. They are additional trainable weight components inserted into selected linear layers of the starting model.

Conceptually, LoRA works like this:

- the original model weights stay frozen,
- small low-rank matrices are added to chosen layers,
- training updates only those smaller matrices,
- at inference time, the model behaves like the original model plus the learned adapter behavior.

This matters for three reasons:

1. training becomes much cheaper than full fine-tuning,
2. the saved artifact is much smaller than the full model,
3. multiple adapters can be trained for different domains while reusing the same starting model.

In this lab, the adapter learns patterns from `MediCore.json`, such as the style of short factual answers and the domain-specific hospital information.

> [!NOTE]
> LoRA does not mean the model is learning only new facts. It is also learning how those facts are presented. If `MediCore.json` contains short direct answers, the adapter will tend to reproduce that answer style.

### Key snippet

```python
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

### Why this matters

- `r` controls adapter capacity,
- `lora_alpha` scales adapter influence,
- `target_modules` defines where adapters are inserted,
- `bias="none"` avoids training bias parameters.

The low-rank idea in LoRA comes from approximating weight updates with smaller matrices rather than full-size dense updates. In practical terms, you can think of `r` as controlling how much expressive capacity the adapter has.

- lower `r` means smaller adapters and lower memory use,
- higher `r` means more capacity but greater memory cost and a higher risk of overfitting on a small dataset.

> [!IMPORTANT]
> The exact `target_modules` are model-architecture specific. Do not assume these names work for every model family.

> [!TIP]
> If you switch to another model architecture, this is one of the first places to verify.

---

## 9. Training setup and memory-saving choices

The training cell combines multiple ideas at once:

- batching,
- validation split,
- learning rate,
- gradient accumulation,
- mixed precision,
- gradient checkpointing.

### Key snippet

```python
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
```

### Why these settings matter

- `mlm=False` is required because this is next-token generation, not masked-language modeling.
- `per_device_train_batch_size=1` reduces memory usage.
- `gradient_accumulation_steps=2` preserves a larger effective batch size.
- `fp16=True` reduces VRAM use.
- `gradient_checkpointing_enable()` saves memory at the cost of some speed.

> [!IMPORTANT]
> `per_device_train_batch_size` and `gradient_accumulation_steps` work together.

If batch size must be reduced because of memory errors, gradient accumulation is often kept or increased to preserve training stability.

---

## 10. Overfitting in small datasets

This lab uses a relatively small dataset. That makes overfitting likely.

Typical signs:

- training loss keeps decreasing,
- validation loss increases early,
- generated answers become too rigid or memorized.

**IMPORTANT:** A lower training loss does not automatically mean a better model.

### Practical actions if overfitting appears

- reduce `num_train_epochs`,
- test the model with several unseen prompts,
- inspect output quality, not just the plotted or printed loss,
- keep the validation split instead of training on everything immediately.

---

## 11. Saving and reloading the adapter

After training, the lab saves the adapter and tokenizer.

### Key idea

When LoRA is used, `trainer.save_model(...)` does not save the whole starting-model checkpoint. It saves the adapter weights.

That is why reloading later requires:

1. loading the original starting-model checkpoint,
2. loading the adapter on top of it.

### Key snippet

```python
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

**IMPORTANT:** Re-enabling cache is useful at inference time, even though it had to be disabled during training.

---

## 12. Inference after fine-tuning

Inference should use the same message structure that the model saw during training.

### Key snippet

```python
messages = [
	{"role": "user", "content": "Who leads the neurology department at MediCore Hospital?"}
]

text = tokenizer.apply_chat_template(
	messages,
	tokenize=False,
	add_generation_prompt=True
)
```

### Why `add_generation_prompt=True` matters

During inference, the model needs a prompt that signals:

`The user has finished. Now the assistant should respond.`

That is why inference uses `add_generation_prompt=True`, while training used `False`.

> [!IMPORTANT]
> If training used a system prompt or JSON-specific assistant behavior, inference should include that same structure again.

---

## 13. Structured output is part of the project workflow

For this course, structured output is not a side topic. It is part of the core workflow because the mini project requires you to connect the model to Gradio.

Two ideas should be separated clearly:

- how the model generates an answer,
- how that answer is packaged for downstream use.

### Recommended project approach

For the project, the most reliable beginner-friendly approach is:

1. generate plain text from the model,
2. wrap that output in JSON using Python.

This is more reliable than depending only on the model to always produce valid JSON.

### Key snippet

```python
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

### Why this path is recommended

- it guarantees valid JSON,
- it is easier to debug,
- it is easier to adapt to your own output schema,
- it integrates cleanly with Gradio.

> [!IMPORTANT]
> If the project expects a different JSON schema, change the wrapper keys here.

---

## 14. Gradio as the application layer

Gradio is the interface layer that lets you turn the model into a simple interactive app.

### Key snippet

```python
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

### What to take from this step

- the fine-tuned model is not the whole project,
- the interface matters,
- structured output makes it easier to display or parse results.

> [!IMPORTANT]
> Gradio and JSON belong to the project path here. They are not unrelated extras.

---

## 15. Exact cells you will modify for your own dataset

This should be explicit because Activity 1 is intended to be reused for the mini project.

### Must change

- the dataset download cell before Step 3 if they are not using [`MediCore.json`](./MediCore.json),
- Cell 2b baseline prompt,
- Cell 3 dataset filename and `preprocess(sample)` mapping,
- Cell 7 test prompt,
- the JSON schema cell used for Gradio output,
- the Gradio interface text and labels.

### May change

- Cell 4 LoRA settings if model family changes,
- Cell 5 training hyperparameters if dataset size or VRAM constraints change.

### The most important customization point

> [!IMPORTANT]
> Cell 3 is the main customization point.

You usually do not need to rewrite the overall pipeline. You need to adapt the mapping from your dataset into the required chat message structure.

---

## 16. Common failure points

### Failure point 1: wrong dataset mapping

Symptoms:

- training runs, but answers are poor,
- the model answers in the wrong style,
- the model ignores the intended task.

Cause:

- incorrect `prompt`/`completion` mapping,
- inconsistent formatting across rows,
- training examples not matching the target use case.

### Failure point 2: training and inference mismatch

Symptoms:

- model output looks unstable,
- JSON format breaks,
- responses ignore system instructions.

Cause:

- different message structures between training and inference.

### Failure point 3: memory problems in Colab

Symptoms:

- CUDA out-of-memory errors,
- notebook crashes or restarts.

Actions:

- reduce `per_device_train_batch_size`,
- keep or increase `gradient_accumulation_steps`,
- avoid unnecessarily long examples,
- keep `fp16=True`.

### Failure point 4: relying only on loss values

Symptoms:

- training seems successful numerically,
- but outputs are low quality or too rigid.

Action:

- evaluate with multiple prompts from the actual task.

---

## 17. What is required, recommended, and optional

### Required for the mini project

- correct chat-template preprocessing,
- LoRA fine-tuning,
- testing with consistent prompt structure,
- structured JSON output,
- Gradio-based interface or rendering.

### Recommended

- Hugging Face authentication,
- baseline inference before training,
- trying several evaluation prompts,
- saving a merged model if deployment needs it.

### Optional or appendix-level material

- alternative JSON-enforcement strategies,
- advanced Pydantic schemas,
- QLoRA and larger-model discussion,
- fine-tuning versus RAG comparison.

---

## 18. Fine-tuning versus RAG

Fine-tuning and Retrieval-Augmented Generation (RAG) solve different problems.

### Fine-tuning is best when:

- you want to adapt behavior,
- you want a specific answer style or format,
- the dataset is small and focused,
- the goal is a self-contained adapted model.

### RAG is best when:

- facts change often,
- the knowledge base is large,
- answers must stay grounded in live documents,
- retraining would be inconvenient or unnecessary.

> [!NOTE]
> It is entirely possible that RAG would be more suitable for some mini-project ideas. For example, if the information changes frequently or comes from many documents, retrieving that information at runtime may be a better design than baking it into model behavior through fine-tuning.

> [!TIP]
> The reason this module focuses on fine-tuning is pedagogical. RAG can be studied independently, but fine-tuning gives direct experience with dataset design, prompt structure, adapter-based training, and model adaptation. Those ideas are useful even when you later decide that RAG is the better production choice.

### A practical distinction

- Fine-tuning changes model behavior by updating adapter weights.
- RAG leaves model weights unchanged and retrieves external information at inference time.

With `MediCore.json`, fine-tuning is a reasonable teaching example because the dataset is small, focused, and easy to map into a prompt-response format.

---

## 19. Recap

The most important ideas from this part are:

1. fine-tuning is not just about changing model weights, but also about teaching a pattern of input-output formatting,
2. dataset mapping is the most important customization step,
3. training and inference prompt structures must stay aligned,
4. LoRA makes fine-tuning practical on limited hardware,
5. JSON output and Gradio are part of the core project workflow,
6. evaluate with real prompts, not only loss values,
7. depending on the problem, RAG may later prove more suitable than fine-tuning.

Activity 1 provides the full hands-on workflow. This part is meant to clarify why each step exists before you modify the lab for your own project.


---

## Links

- [AI Engineering (Chapter 7. Finetuning), by Chip Huyen](https://metropolia.finna.fi/Record/nelli15.36974248300041)
- Course: [fine-tuning techniques for large language models.](https://huggingface.co/learn/smol-course/unit0/1)