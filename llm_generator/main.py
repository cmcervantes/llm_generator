import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from pprint import pprint

qa_dataset = load_dataset("meta-math/MetaMathQA", split="train[:50]")
qa_dataset = qa_dataset.train_test_split(test_size=0.1)

#model_id = "mistralai/Mixtral-8x7B-v0.1"
model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

#text = "Hello my name is"
#inputs = tokenizer(text, return_tensors="pt").to(0)

random_inputs_orig =


pprint(tokenized_qa["train"].with_format("torch")[0])


model = AutoModelForCausalLM.from_pretrained(
    model_id, load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

single_input = tokenized_qa["train"].with_format("torch")[0]

outputs = model.generate(**tokenized_qa["train"].with_format("torch")[0], max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
