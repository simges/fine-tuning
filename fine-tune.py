# def format_prompts(examples):
#    """
#    Define the format for your dataset
#    This function should return a dictionary with a 'text' key containing the formatted prompts
#    """
#    pass

from peft import prepare_model_for_kbit_training

def format_prompts(batch):
    return {
            "question": [f"Q: {q}" for q in batch["question"]],  # List
            "answer": batch["answer"]                           # List (unchanged)
        }

from datasets import load_dataset

# dataset = load_dataset("your_dataset_name", split="train")
dataset = load_dataset("b-mc2/sql-create-context", split="train")
dataset = dataset.map(format_prompts, batched=True)

dataset['question'][2] # Check to see if the fields were formatted correctly


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="your_model_name",
    num_train_epochs=4, # replace this, depending on your dataset
    per_device_train_batch_size=16,
    learning_rate=1e-5,
    optim="sgd"
)

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    dataset_text_field='question',
    max_seq_length=1024,
)

#trainer.train()

trainer.train(resume_from_checkpoint="your_model_name/checkpoint-19648")
adapter_model = trainer.model
merged_model = adapter_model.merge_and_unload()

trained_tokenizer = trainer.tokenizer

from huggingface_hub import login
login() # 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

from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id="super-cool-model")

repo_id = "super-cool-model"

merged_model.push_to_hub(repo_id)
trained_tokenizer.push_to_hub(repo_id)

