# This code prepares the diplomat dataset in prompts and tokenizes them for fine-tuning

repo_name = 'TizianoGaddo/diplomat-tokenized' # name of the repo on huggingface where the tokenized dataset will be saved
# Please, change the huggingface repo name to the name of your repo if running the code again

from huggingface_hub import login
login(token="your-huggingface-token")
model_name = "TizianoGaddo/gpt-bert-base"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

from datasets import load_dataset
dataset_cqa = load_dataset("henry12348/DiPlomat", "CQA", trust_remote_code=True, streaming=False)

# Prepares prompts
def gen_text_diplomat(examples):
    examples["prompt"] = [
        "CONTESTO:\n" +
        "\n".join([f"{speaker}: {turn}" for speaker, turn in zip(dialogue_speakers, dialogue_text)]) +
        f"\nDOMANDA\n{question}\nRISPOSTA\n{answer}"
        for dialogue_text, dialogue_speakers, question, answer in zip(
            examples["text"],
            examples["speaker"],
            examples["questions"],
            examples["answer"]
        )
    ]
    return examples

# Tokenizes prompt
def tokenize_func(example):
  tokenized = tokenizer(example['prompt'], padding='max_length', truncation=True, max_length=512)
  tokenized['labels'] = tokenized['input_ids'].copy()
  return tokenized

# Mapping previous two functions to the dataset and saves the tokenized dataset on huggingface
diplomat = dataset_cqa.map(gen_text_diplomat, batched=True)
tokenized_diplomat = diplomat.map(tokenize_func, batched = True, remove_columns=["text", "speaker", "gold_statement", "questions", "answer"])
tokenized_diplomat.push_to_hub(repo_name)

# Counts total words in the prompts created from the diplomat dataset
words = []
for example in (diplomat['train']['prompt']):
  words.append(len(example.split()))

total_words = sum(words)