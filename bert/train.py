import os

import torch
from datasets import load_from_disk, load_metric
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, TrainingArguments, Trainer

os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        
        self.dataset = load_from_disk('../Dataset/data/ChnSentiCorp')
        self.dataset = self.dataset.map(tokenize_func, batched=True)
        self.dataset = self.dataset[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def tokenize_func(examples):
    return tokenizer(examples['text'],
                     truncation=True,
                     padding='max_length')


def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    #编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)

    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    #print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels

datasets = load_from_disk('../Huggingface_Toturials/data/ChnSentiCorp')
tokenized = datasets.map(tokenize_func, batched=True)
train_dataset = tokenized['train']
test_dataset = tokenized['test']
valid_dataset = tokenized['validation']
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=8)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=8)

model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
trainning_args = TrainingArguments(
    output_dir='./outputs',
    evaluation_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=trainning_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

metrics = trainer.evaluate()
predictions = trainer.predict(test_dataset)