repo_name = "TizianoGaddo/gpt-bert-finetuned-diplomat" # name of the repo on huggingface where the model will be saved
# Please, change the name of the huggingface repo to your repo if running the code again

model_name = "TizianoGaddo/gpt-bert-base"

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from huggingface_hub import login
login(token="your-huggingface-token")
import wandb
wandb.init(mode="disabled")

from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model.to(device)

from datasets import load_dataset
tokenized_dataset = load_dataset('TizianoGaddo/diplomat-tokenized', trust_remote_code = True)
tokenized_dataset.set_format(type='torch')

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir = repo_name,
    eval_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    dataloader_num_workers = 8,
    save_safetensors=False,
    logging_strategy='epoch',
    save_strategy="epoch",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
    )

trainer.train()
trainer.push_to_hub()
tokenizer.push_to_hub(repo_name)

print("All done!")