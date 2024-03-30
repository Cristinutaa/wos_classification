print("Importing packages...")
import argparse
import datasets
import numpy as np
import pandas as pd
import os
import json
import random
import sys
from ast import literal_eval

import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import tensorflow as tf
import transformers
from transformers import (Trainer, AutoTokenizer, AutoConfig, 
    set_seed, AutoModelForSequenceClassification,default_data_collator, 
    EvalPrediction, AdamW, get_linear_schedule_with_warmup, TrainingArguments,EarlyStoppingCallback)

from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from datasets import load_dataset
import pickle
#________________________________________________________________________________________________
# class for a multi-label dataset
class ML_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype= torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# compute metrics function for HF Trainer class
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    preds = torch.sigmoid(torch.from_numpy(preds)).numpy()
    # top3index = np.argpartition(preds, -3)[-3:]
    # preds.fill(0)
    # preds[top3index] = 1
    threshold = np.partition(preds, -3)[-3]
    preds[preds<threshold] = 0
    preds[preds>=threshold] = 1
    preds = preds.tolist()

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main(args):
    # load training data
    # args.label_col is the label column in your training data
    train = pd.read_csv(args.train_path, index_col = False).dropna(subset=['text', args.label_col]).drop_duplicates(subset=['text', args.label_col])
    test = pd.read_csv(args.test_path, index_col = False).dropna(subset=['text', args.label_col]).drop_duplicates(subset=['text', args.label_col])

    train['source'] = ['train']*len(train)
    test['source'] = ['test']*len(test)

 

    df = pd.concat([train, test])
    print("Columns in df:", df.columns)

    df['text'] = df[args.text_col]
    df['labels'] = df[args.label_col]
    df = df.dropna(subset=['text', 'labels']) # remove empty rows

    df.labels = df.labels.apply(literal_eval) # labels need to be either a list or tuple of individual labels per entry, e.g. ['class1', 'class2'], ['class1']
    
    mlb = MultiLabelBinarizer() # transform list of labels to binary labels [0, 1, 1, 0]
    df.labels = [i for i in mlb.fit_transform(df.labels)]

    train = df[df['source']=='train']
    test = df[df['source']=='test']


    train_texts = train.text.tolist()
    train_labels = train.labels.tolist()

    test_texts = test.text.tolist()
    test_labels = test.labels.tolist()

#__________________________________________________________________________________________
    #prepare datasets
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_encodings = tokenizer(train_texts,
                max_length = args.max_length,
                padding = 'max_length',
                truncation=True,
                return_token_type_ids=False)
    train_dataset = ML_Dataset(train_encodings, train_labels)

    
    test_encodings = tokenizer(test_texts,
                max_length = args.max_length,
                padding = 'max_length',
                truncation=True,
                return_token_type_ids=False)
    test_dataset = ML_Dataset(test_encodings, test_labels)
#________________________________________________________________________________________________________________
    #prepare model
    num_labels = int(len(mlb.classes_))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        problem_type="multi_label_classification",
        num_labels=num_labels
        )
#________________________________________________________________________________________________________________
    #prepare Trainer
    training_args = TrainingArguments(
            output_dir = args.output_dir,
            overwrite_output_dir = True,
            do_train=True,
            do_eval=True,
            do_predict=True,
            load_best_model_at_end = True,

            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.test_batch_size,
            gradient_accumulation_steps=args.accumulate_grad,
            learning_rate=args.learning_rate,

            num_train_epochs = args.epochs,
            lr_scheduler_type= 'linear',

            logging_strategy='epoch',
            save_strategy=args.save_strategy,
            evaluation_strategy='epoch',
            seed=args.seed,
            fp16=args.fp16,
           #n_gpu = 1,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if args.do_predict else None,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.es_patience)]
    )
    #train model
    train_result = trainer.train()
#__________________________________________________________________________________________________________
    # save finetuned model
    tokenizer.save_pretrained(os.path.join(args.output_dir,"finetuned_model"))
    trainer.save_model(os.path.join(args.output_dir,"finetuned_model"))
    trainer.save_state()
    trainer.save_metrics("train_results.json", metrics=train_result.metrics)

    # The best model from training (i.e. with the lowest loss on the evaluation set) is used to generate predictions on the test set
    preds = trainer.predict(test_dataset, metric_key_prefix="predict").predictions
    preds = torch.sigmoid(torch.from_numpy(preds)).numpy()
    preds[preds < args.threshold] = 0
    preds[preds >= args.threshold] = 1

    preds = preds.tolist()

    # get classification report
    report = classification_report(test_labels, preds, digits=4, target_names=mlb.classes_)
    print(report)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    info_dict = {"true_labels": test_labels,
                 "predictions": preds,
                 "labels_name": mlb.classes_,
                 "mlb_model": mlb}
    
    with open(os.path.join(args.output_dir,f'{args.outfile_name}.pickle'), 'wb') as outfile:
        pickle.dump(info_dict, outfile)

    precision, recall, f1, support = precision_recall_fscore_support(test_labels, preds, average='macro', pos_label=1)
    emr = accuracy_score(test_labels, preds)

    results_dict = {
        "F1-macro": f1,
        "Precision": precision,
        "Recall": recall,
        "Exact match ratio": emr
    }

    with open(os.path.join(args.output_dir,'test_results.json'), 'w') as outfile:
        json.dump(results_dict, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--label_col', type=str, default='labels')
    parser.add_argument('--text_col', type=str, default='text')

    parser.add_argument('--model_name', type=str, default='wietsedv/bert-base-dutch-cased')
    parser.add_argument('--train_batch_size', default= 8, type=int)
    parser.add_argument('--test_batch_size', default= 8, type=int)

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--learning_rate', default = 2e-5, type =float)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--accumulate_grad', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=int, default=1.)
    parser.add_argument('--es_patience', type=int, default=3)

    parser.add_argument('--do_predict', type=bool, default=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    
    parser.add_argument('--fp16', type=bool, default=False)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_strategy', type=str, default='epoch')

    parser.add_argument('--output_dir', type=str, default='outputs/')
    parser.add_argument('--outfile_name', type=str, default= 'predictions.json')
    parser.add_argument('--finetuned_model_name', type=str, default= 'Finetuned_model')

    args = parser.parse_args()

    try:
        os.mkdir(args.output_dir)
    except:
        pass

    #save arguments passed to the script (simplifies back-tracking and debugging!)    
    with open(os.path.join(args.output_dir,'arguments.txt'), 'w') as f:
        f.write(str(args))


    main(args)