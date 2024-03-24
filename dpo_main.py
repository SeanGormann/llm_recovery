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
from sentence_transformers import SentenceTransformer, util
import torch
from time import time
#from sklearn.metrics.pairwise import cosine_similarity

total_memory = torch.cuda.get_device_properties(0).total_memory
free_memory = total_memory - torch.cuda.memory_allocated(0)
print(f"Total GPU Memory: {total_memory / 1e9} GB, Free Memory: {free_memory / 1e9} GB")



#import flash-attention


model_path = "google/gemma-2b-it" # "google/gemma-2b-it" "microsoft/phi-2"
access_token = "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"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

device_map = "auto"

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    device_map=device_map,
    trust_remote_code=True,
    token=access_token
)

base_model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=access_token) #microsoft/phi-2

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
output_dir = "/llm_recovery/"



dataset_path = '/llm_recovery/data_generation/dpo_dataset_v1.json'
dataset = load_dataset('json', data_files=dataset_path)
print(dataset)

"""
def create_custom_prompt(tokenizer, original_text, rewritten_text, max_tokens=512):
    task_description = "Determine what rewrite prompt was used to convert this text."
    assistant_message = "Please provide the Original_Text and Rewritten_Text."
    
    # Function to truncate text based on token count and note the original length if truncated
    def truncate_text(text, max_tok):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > max_tok:
            truncated_tokens = tokens[:max_tok]
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            return truncated_text + " ... [Text truncated]", len(tokens)
        return text, len(tokens)

    truncated_original, original_tokens = truncate_text(original_text, max_tokens)
    truncated_rewritten, rewritten_tokens = truncate_text(rewritten_text, max_tokens)
    
    # Constructing the user content with both original and rewritten texts, including token counts
    user_content = f"Original_Text: {truncated_original} (Original tokens: {original_tokens}) , Rewritten_Text: {truncated_rewritten} (Rewritten tokens: {rewritten_tokens})"
    
    chat = [
        { "role": "user", "content": task_description },
        { "role": "assistant", "content": assistant_message },
        { "role": "user", "content": user_content },
    ]

    return chat
"""

# Load a sentence transformer model for embedding calculation
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def evaluate_model_with_similarity(test_dataset, tokenizer, model, num_samples=10):
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Move model to the appropriate device
    #model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    results = []

    for i in range(num_samples):
        # Extract a single test sample
        test_sample = test_dataset[i]
        
        # Assuming 'test_sample' contains 'chosen' which we compare with the output
        chosen_text = test_sample['chosen']
        """
        prompt = create_custom_prompt(
            tokenizer,
            test_sample['original_text'],
            test_sample['rewritten_text']
        )
        """
        inputs = tokenizer.encode(test_sample["prompt"], add_special_tokens=False, return_tensors="pt")
        input_length = inputs.shape[1]

        start_time = time()

        outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)
        new_tokens = outputs[0, input_length:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        end_time = time()
        time_taken = end_time - start_time

        # Compute embeddings for both chosen and generated text
        chosen_embedding = sentence_model.encode(chosen_text, convert_to_tensor=True)
        generated_embedding = sentence_model.encode(generated_text, convert_to_tensor=True)

        # Compute Cosine similarity
        cosine_similarity = util.cos_sim(chosen_embedding, generated_embedding).item()
        
        #similarity_scores = cosine_similarity(prompt_embeddings, prompt_1_embeddings)
        #similarity_scores = np.diag(similarity_scores)

        results.append({
            'prompt': test_sample["prompt"],
            'chosen': chosen_text,
            'output': generated_text,
            'time_taken': time_taken,
            'cosine_similarity': cosine_similarity
        })
    
    return results

# Example usage
results = evaluate_model_with_similarity(dataset["train"], tokenizer, base_model, num_samples=5)

for result in results:
    #print("Prompt:", result['prompt'])
    print("Chosen:", result['chosen'])
    print("Output:", result['output'])
    print("Time taken:", result['time_taken'], "seconds")
    print("Cosine similarity:", result['cosine_similarity'], "\n")



# from https://github.com/mlabonne/llm-course/blob/main/Fine_tune_a_Mistral_7b_model_with_DPO.ipynb
lora_dropout=0.05
lora_alpha=16
lora_r=16
learning_rate=5e-5 # 5e-4
batch_size = 4

def create_peft_config(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        lora_dropout=lora_dropout,
        lora_alpha=lora_alpha,
        r=lora_r,
        bias="none",
        target_modules = ["q_proj", "k_proj", "v_proj"] #, "o_proj", "gate_proj"] #"up_proj", "down_proj"
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


trainer = DPOTrainer(
    model, # model base_model
    ref_model=None,
    args=training_args,
    train_dataset=dataset["train"], # test_dataset dataset
    #test_dataset=dataset["test"],
    tokenizer=tokenizer,
    peft_config=lora_config,
    beta=0.1,
    #max_prompt_length=2048, #changed from 1024
    #max_length=1536,
)


print("Starting trainer...")
trainer.train()

# todo: during training getting these warning:

# i guess this is on the base model, need to check. in that case this is fine
# UserWarning: None of the inputs have requires_grad=True. Gradients will be None

# seems that this can be ignored:
# Could not estimate the number of tokens of the input, floating-point operations will not be computed
model_name = "gemma_2b_it"
output_dir = os.path.join(output_dir, f"final_checkpoint_{model_name}")
trainer.model.save_pretrained(output_dir)

results = evaluate_model_with_similarity(dataset["train"], tokenizer, model, num_samples=5)


