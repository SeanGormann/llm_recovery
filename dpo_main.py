import torch
import bitsandbytes
import accelerate
import transformers
import optimum
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import DPOTrainer
import pandas as pd
from datasets import load_dataset
import os



#torch.set_default_device("cuda")

#model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# load the training dataset
#dataset = load_dataset("json", data_files={'train': dataset_file})
#dataset = pd.read_csv("/llm_recovery/data_generation/dpo_dataset_v1.csv")
#dataset = dataset['train'].shuffle(seed=42)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

device_map = "auto"

base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    #"mistralai/Mistral-7B-Instruct-v0.1",
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
)
base_model.config.use_cache = False

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
output_dir = "/llm_recovery/"

"""
# Assuming `df` is your DataFrame
dataset.to_json("your_dataset.json", orient="records", lines=True)
# Load the training dataset
dataset_file = "your_dataset.json"
dataset = load_dataset("json", data_files={'train': dataset_file})
dataset = dataset['train'].shuffle(seed=42)

def truncate_text(example, max_length=900):
    # Tokenize the original and rewritten texts to check their length
    tokens_original = tokenizer.encode(example['original_text'], add_special_tokens=False)
    tokens_rewritten = tokenizer.encode(example['rewritten_text'], add_special_tokens=False)
    #print(len(tokens_original), len(tokens_rewritten))

    # Check if the length exceeds max_length and truncate if necessary
    if len(tokens_original) > max_length:
        # Decode back to text after truncating
        example['original_text'] = tokenizer.decode(tokens_original[:max_length], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    if len(tokens_rewritten) > max_length:
        # Decode back to text after truncating
        example['rewritten_text'] = tokenizer.decode(tokens_rewritten[:max_length], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return example


def get_prompt(example):
    # Assuming `tokenizer` and other necessary components are defined elsewhere in your notebook
    example = truncate_text(example)

    prompt_sample = [
        {"role": "system", "content": "From the given original and rewritten texts, predict the rewrite prompt used to transform the original text."},
        {"role": "user", "content": f"Original: {example['original_text']} ----- Rewritten: {example['rewritten_text']} "}
    ]
    prompt_for_model = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
    example['prompt'] = prompt_for_model

    example['chosen'] = example['chosen_prompt'] + tokenizer.eos_token
    example['rejected'] = example['rejected_prompt'] + tokenizer.eos_token

    return example

print("Mapping Dataset..")
# Map the function over the dataset
dataset = dataset.map(get_prompt)
dataset = dataset.rename_column("chosen_score", "score_chosen")
dataset = dataset.rename_column("rejected_score", "score_rejected")



##Testing
# Define the system message
example = dataset[0]
input_text = example["prompt"]

# Tokenize input text, ensuring to generate an attention mask this time
inputs = tokenizer(input_text, return_tensors="pt", max_length=2000, truncation=True, padding=True)

# For open-ended generation, setting pad_token_id explicitly if your model does not have one set
if base_model.config.pad_token_id is None:
    base_model.config.pad_token_id = base_model.config.eos_token_id

# Generate output using the updated inputs
outputs = base_model.generate(**inputs, max_length=2000, pad_token_id=base_model.config.pad_token_id)
text = tokenizer.batch_decode(outputs)[0]

# Output the model's generated text and the actual rewrite prompt for comparison
actual_rewrite_prompt = example['chosen_prompt']
print("Generated Prompt:", text)
print("Actual Rewrite Prompt:", actual_rewrite_prompt)

"""

dataset = load_dataset("unalignment/toxic-dpo-v0.2")
print(dataset)

def preprocess_function(examples):
    # Tokenize the 'prompt', 'chosen', and 'rejected' fields and truncate them if necessary
    prompt_encodings = tokenizer(examples['prompt'], truncation=True, max_length=2048, padding="max_length", return_tensors="pt")
    chosen_encodings = tokenizer(examples['chosen'], truncation=True, max_length=2048, padding="max_length", return_tensors="pt")
    rejected_encodings = tokenizer(examples['rejected'], truncation=True, max_length=2048, padding="max_length", return_tensors="pt")
    
    # Ensure we return a list of texts, not token IDs, after truncation
    truncated_prompts = [tokenizer.decode(enc, skip_special_tokens=True, clean_up_tokenization_spaces=True) for enc in prompt_encodings.input_ids]
    truncated_chosens = [tokenizer.decode(enc, skip_special_tokens=True, clean_up_tokenization_spaces=True) for enc in chosen_encodings.input_ids]
    truncated_rejecteds = [tokenizer.decode(enc, skip_special_tokens=True, clean_up_tokenization_spaces=True) for enc in rejected_encodings.input_ids]
    
    # Update the examples with the truncated texts
    examples['prompt'] = truncated_prompts
    examples['chosen'] = truncated_chosens
    examples['rejected'] = truncated_rejecteds

    return examples

# Apply preprocessing to the 'train' split
dataset.map(preprocess_function, batched=True)



# from https://github.com/mlabonne/llm-course/blob/main/Fine_tune_a_Mistral_7b_model_with_DPO.ipynb
lora_dropout=0.05
lora_alpha=16
lora_r=16
learning_rate=5e-4 # 5e-4
batch_size = 4

def create_peft_config(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        lora_dropout=lora_dropout,
        lora_alpha=lora_alpha,
        r=lora_r,
        bias="none",
        target_modules = ["q_proj", "k_proj", "v_proj"] #, "o_proj", "gate_proj"]
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    return model, peft_config

model, lora_config = create_peft_config(base_model)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,

    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    warmup_steps=50,
    logging_steps=1,
    num_train_epochs=1,
    save_steps=50,
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
)

from datasets import load_dataset

dataset_file = "dpo_dataset.json"
"""
 # Load the dataset
#dataset = load_dataset("json", data_files="dpo_dataset.json", field="rows")
#dataset = load_dataset("Anthropic/hh-rlhf")
dataset = load_dataset("argilla/dpo-mix-7k")
print(dataset)

def construct_prompt(example):
    # Extract the question and initial answer from the 'chosen' field (assuming it's similar for 'rejected')
    question = example['chosen'][0]['content']
    initial_answer = example['chosen'][1]['content']
    
    # Extract the completions from both 'chosen' and 'rejected' fields
    chosen_completion = example['chosen'][1]['content']
    rejected_completion = example['rejected'][1]['content']
    
    # Construct the prompt
    prompt = f"Question and Initial Answer:\n{question}\n\nChosen Completion:\n{chosen_completion}\n\nRejected Completion:\n{rejected_completion}\n\n"
    prompt += "Based on the initial answer, which completion is more accurate and relevant? Choose 'Chosen' or 'Rejected'."
    
    example['prompt'] = prompt
    return example


dataset = dataset.map(construct_prompt)
print(dataset)
"""

trainer = DPOTrainer(
    model, # model base_model
    ref_model=None,
    args=training_args,
    train_dataset=dataset["train"], # test_dataset dataset
    tokenizer=tokenizer,
    peft_config=lora_config,
    beta=0.1,
    max_prompt_length=2048, #changed from 1024
    max_length=1536,
)


# Use this generator in your DataLoader, Sampler, or other components that require random operations


print("Starting trainer...")
trainer.train()

# todo: during training getting these warning:

# i guess this is on the base model, need to check. in that case this is fine
# UserWarning: None of the inputs have requires_grad=True. Gradients will be None

# seems that this can be ignored:
# Could not estimate the number of tokens of the input, floating-point operations will not be computed

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)


##Testing
# Define the system message
example = dataset["train"][0]
input_text = example["prompt"]

# Tokenize input text, ensuring to generate an attention mask this time
inputs = tokenizer(input_text, return_tensors="pt", max_length=2000, truncation=True, padding=True)

# For open-ended generation, setting pad_token_id explicitly if your model does not have one set
if model.config.pad_token_id is None:
    model.config.pad_token_id = base_model.config.eos_token_id

# Generate output using the updated inputs
outputs = model.generate(**inputs, max_length=2000, pad_token_id=model.config.pad_token_id)
text = tokenizer.batch_decode(outputs)[0]

# Output the model's generated text and the actual rewrite prompt for comparison
actual_rewrite_prompt = example['chosen']
print("Generated Prompt:", text)
print("Actual Rewrite Prompt:", actual_rewrite_prompt)


