# Generated from: fine-tune.ipynb
# Converted at: 2026-06-26T15:49:52.005Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from huggingface_hub import notebook_login

notebook_login()

!pip install -U transformers datasets peft accelerate bitsandbytes --quiet

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import torch

model_path = "google/gemma-3-1b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)


import torch
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from datasets import Dataset
import json

with open("/content/train.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

dataset = Dataset.from_list(data)

for i in range(min(3, len(dataset))):
    print(json.dumps(dataset[i], indent=2, ensure_ascii=False))

from transformers import DataCollatorForLanguageModeling

def tokenize(example):
    return tokenizer(
        tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        ),
        truncation=True,
        max_length=4096,
    )

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

lengths = [
    len(tokenizer.apply_chat_template(x["messages"], tokenize=True))
    for x in dataset
]
print("Max length:", max(lengths))
print("Avg length:", sum(lengths) // len(lengths))
print(tokenizer.apply_chat_template(dataset[0]["messages"], tokenize=False))

torch.cuda.empty_cache()

from transformers import TrainingArguments, Trainer

model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    warmup_ratio=0.1,
    learning_rate=2e-4,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_dir="logs",
    logging_steps=50,
    save_steps=250,
    save_total_limit=1,
    report_to="none",
    fp16=True,
    group_by_length=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager",
    trust_remote_code=True
)

lora_model = PeftModel.from_pretrained(
    base_model,
    "/content/drive/MyDrive/output/checkpoint-525",
    is_trainable=False
)

merged_model = lora_model.merge_and_unload()

save_path = "/content/drive/MyDrive/Sorachio-1B-4096-2e-4"
merged_model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/content/drive/MyDrive/Sorachio-1B-4096-2e-4"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager"
).eval()

messages = [
    {"role": "user", "content": "Perkenalkan dirimu"}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=(input_ids != tokenizer.pad_token_id).long(),
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=0.6,
        pad_token_id=tokenizer.eos_token_id
    )

output_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(output_text)