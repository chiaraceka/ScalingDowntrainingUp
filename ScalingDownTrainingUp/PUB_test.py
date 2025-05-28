from huggingface_hub import login
login(token="your-huggingface-token")

# choose the model on which to run the test
def choose_model():
  yes_no = input('Is the model finetuned? (y/n) ')
  if yes_no.lower() in ['y', 'yes']:
    return("TizianoGaddo/gpt-bert-finetuned-diplomat")
  elif yes_no.lower() in ['n', 'no']:
    return("TizianoGaddo/gpt-bert-base") # this is a copy of the original bert base with only the temperature changed (to None, because otherwise it will give error during fine-tuning)
  else:
    return(print('Error: invalid input.\nPlease enter "y" or "n".'))

model_name = choose_model()

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.to(device)

import pandas as pd
import datasets
from datasets import load_dataset
# import all the tasks of the PUB
task1  = load_dataset("TizianoGaddo/SDTU", "1", trust_remote_code=True)['train']
task2  = load_dataset("TizianoGaddo/SDTU", "2", trust_remote_code=True)['train']
task3  = load_dataset("TizianoGaddo/SDTU", "3", trust_remote_code=True)['train']
task4  = load_dataset("TizianoGaddo/SDTU", "4", trust_remote_code=True)['train']
task5  = load_dataset("TizianoGaddo/SDTU", "5", trust_remote_code=True)['train']
task6  = load_dataset("TizianoGaddo/SDTU", "6", trust_remote_code=True)['train']
task7  = load_dataset("TizianoGaddo/SDTU", "7", trust_remote_code=True)['train']
task8  = load_dataset("TizianoGaddo/SDTU", "8", trust_remote_code=True)['train']
task9  = load_dataset("TizianoGaddo/SDTU", "9", trust_remote_code=True)['train']
task10 = load_dataset("TizianoGaddo/SDTU", "10", trust_remote_code=True)['train']
task11 = load_dataset("TizianoGaddo/SDTU", "11", trust_remote_code=True)['train']
task12 = load_dataset("TizianoGaddo/SDTU", "12", trust_remote_code=True)['train']
task13 = load_dataset("TizianoGaddo/SDTU", "13", trust_remote_code=True)['train']
task14 = load_dataset("TizianoGaddo/SDTU", "14", trust_remote_code=True)['train']

# Function to generate the PUB prompts (without options)
def gen_question(task_number:int, pretext:str):
  # instructions for each task, taken from the PUB paper
  instructions = ["Your task is to label the 'Response' as an Indirect or Direct answer based on the Context and Question:\n\n",
                  "Your task is to interpret Y's answer to X's question into one of the options:\nA: Yes\nB: No\nC: Yes, subject to some conditions\nD: In the middle, neither yes nor no\nE: Other\n\n",
                  "Your task is to interpret Y's answer to X's question into one of the options:\nA: Yes\nB: No\nC: Yes, subject to some conditions\nD: In the middle, neither yes nor no\nE: Other\n\n",
                  "Your task is to understand the implied meaning in Speaker_2's last response and give the explicit meaning:\n\n",
                  "Your task is to understand the implied meaning in Speaker_2's last response and give the explicit meaning:\n\n",
                  "Your task is to decide if Speaker_2 Agrees or is being Sarcastic with Speaker_1 in the conversation:\n\n",
                  "Your task is to identify the correct meaning of the figurative sentence:\n\n",
                  "Your task is to identify the correct meaning of the figurative sentence from the given hint:\n\n",
                  "Your task is to identify the correct meaning of the figurative sentence from the given hint:\n\n",
                  "",
                  "",
                  "Your task is to deduce if the Assumption is valid or invalid based on the conversation:\n\n",
                  "Your task is to answer the given question based on the conversation:\n\n",
                  "Your task is to answer the Question based on the given Context:\n\n",
                 ]
  ending = "Correct answer:"
  question = instructions[task_number-1] + pretext + '\n' + ending  
  return question                                         

# Funzione che genera la lista di domande
def gen_question_list(dataset:datasets.arrow_dataset.Dataset, task_number:int):
  questions_list = []
  pretext = dataset['pretext']
  for i in range(len(pretext)):                          
    question = gen_question(task_number, pretext[i])    
    questions_list.append(question)                     
  return questions_list


# function that creates a list of indices of the correct answer to each exercise (to compare against the BabyLM answers)
def correct_index(dataset:datasets.arrow_dataset.Dataset):
  result = []
  for i in range(len(dataset['pretext'])):
    try:
        index = dataset['options'][i].index(dataset['correct answer'][i])
    except ValueError:
        index = None
    result.append(index)
  return result


# Function that measures the loss for a string (made up of question + answer)
def score_choice(question, choice):
    full_prompt = f"{question} {choice}"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs[0]
    return loss.item()

# Function that, given a task dataset and the corresponding task number, tests the model on it
def PUB_task_causal(task:datasets.arrow_dataset.Dataset, task_number:int):
  questions = gen_question_list(task, task_number)
  print(f'TASK {task_number}')
  correct_answers = correct_index(task)
  babyLM_answers = []
  print('Progress:\nxxxxxxxxxx')
  for i in range(len(questions)):
    scores = [score_choice(questions[i], choice) for choice in task['options'][i]] # for each answer option, measures the loss of (question+answer option)
    babyLM_answers.append(scores.index(min(scores)))                               # then chooses the most likely answer (i.e. lowest loss)
    if (i!=0 and (i*100)/len(questions) % 10 == 0):
      print('x', end='')
  accuracy=sum([a==b for a,b in zip(babyLM_answers, correct_answers)])/len(babyLM_answers)
  print(f"\nAccuracy: {accuracy}\n\n")
  return accuracy, babyLM_answers, correct_answers

# Function that runs a binomial test on the answers to a task (to check if the model accuracy is significantly greater than chance)
def binomial_pvalue(task_results:pd.DataFrame, task_number:int):
  babyLM_answers = task_results['babyLM_answers']
  correct_answers = task_results['correct_answers']
  n_correct = sum([a==b for a,b in zip(babyLM_answers, correct_answers)])
  n_questions = len(babyLM_answers)
  task_list = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10, task11, task12, task13, task14]
  n_choices = [len(task['options'][0]) for task in task_list]
  prob = 1/n_choices[task_number-1]
  from scipy.stats import binomtest
  results = binomtest(n_correct, n_questions, p=prob, alternative='greater')
  return float(results.pvalue)

# Function that runs the test on all PUB tasks, saves the results of each task, the accuracy and pvalues of the binomial test in csv
def test_PUB():
  yes_no = input('Is the model finetuned? (y/n) ')
  if yes_no.lower() not in ['y', 'yes', 'n', 'no']:
    return(print('Error: invalid input.\nPlease enter "y" or "n".'))
  accuracy_list = []
  pvalues_list = []
  tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10, task11, task12, task13, task14]
  task_names = ["task1", "task2", "task3", "task4", "task5", "task6", "task7", "task8", "task9", "task10", "task11", "task12", "task13", "task14"] 
  import pandas as pd
  for task, task_num in zip(tasks, range(1, 15)):
    task_results = PUB_task_causal(task, task_num)
    task_results_df = pd.DataFrame()
    task_results_df['babyLM_answers'] = task_results[1]
    task_results_df['correct_answers'] = task_results[2]
    if yes_no.lower() in ['y', 'yes']:
      model_type = 'finetuned'
    else:
      model_type = 'base'
    task_results_df.to_csv(f"task{task_num}_{model_type}.csv")
    accuracy_list.append(task_results[0])
    pvalues_list.append(binomial_pvalue(task_results_df, task_num))
  accuracy_df = pd.DataFrame([accuracy_list])
  accuracy_df.columns = task_names
  accuracy_df.to_csv(f'accuracy_{model_type}.csv')
  pvalue_df = pd.DataFrame([pvalue_list])
  pvalue_df.columns = task_names
  pvalue_df.to_csv(f'pvalues_{model_type}.csv')
  return('Done')

test_PUB()