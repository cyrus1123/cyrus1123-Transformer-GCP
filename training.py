

from datasets import load_dataset

dataset = load_dataset("EEG")



from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("big_bird")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)





from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("big_bird", num_labels=5)


from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)


trainer.train('EEGDataTrainingArguments')




del model
del pytorch_model
del trainer
torch.cuda.empty_cache()



from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)



from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)


from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)


import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)



from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # other args and kwargs here
    report_to="wandb",  # enable logging to W&B
    run_name="bert-base-high-lr"  # name of the W&B run (optional)
)

trainer = Trainer(
    # other args and kwargs here
    args=args,  # your training args
)

trainer.train()



metric = load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()



import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as plot 
inpsiz = 784 
hidensiz = 15000
numclases = 10
numepchs = 500
bachsiz = 50
l_r = 0.001 

trainds = torchvision.datasets.EEG(root='./data', 
                                          train=True, 
                                          transform=trans.ToTensor(),  
                                          download=True)
testds = torchvision.datasets.EEG(root='./data', 
                                           train=False, 
                                           transform=trans.ToTensor()) 

trainldr = torch.utils.data.DataLoader(dataset=trainds, 
                                           batch_size=bachsiz, 
                                           shuffle=True)
testldr = torch.utils.data.DataLoader(dataset=testds, 
                                           batch_size=bachsiz, 
                                           shuffle=False)

class neural_network(nn.Module):
    def __init__(self, inpsiz, hidensiz, numclases):
         super(neural_network, self).__init__()
         self.inputsiz = inpsiz
         self.l1 = nn.Linear(inpsiz, hidensiz) 
         self.relu = nn.ReLU()
         self.l2 = nn.Linear(hidensiz, numclases) 
    def forward(self, y):
         outp = self.l1(y)
         outp = self.relu(outp)
         outp = self.l2(outp)

         return outp
modl = neural_network(inpsiz, hidensiz, numclases)
class Preprop():
  
  def preprop(self, args):
    criter = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(modl.parameters(), lr=l_r)
    nttlstps = len(trainldr)
    for epoch in range(numepchs):
        for x, (item, lbls) in enumerate(trainldr): 
            item = item.reshape(-1, 28*28)
            labls = lbls

            outp = modl(item)
            losses = criter(outp, lbls)

            optim.zero_grad()
            losses.backward()
            optim.step() 
