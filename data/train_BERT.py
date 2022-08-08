import argparse
import pandas as pd
import torch
import numpy as np
import time
import datetime
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

# add base path so that can import other files from project
import os
import sys
LIB_DIR = os.path.abspath(__file__).split('data')[0]
sys.path.insert(1, LIB_DIR)

from src.Attacks.utils.optim_utils import seed_everything


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def pad_sequences(input_ids, maxlen):
    padded = []
    for inp in input_ids:
        if len(inp) >= maxlen:
            padded.append(inp[:maxlen-1] + [inp[-1]])
        else:
            padded.append(inp + [0]*(maxlen - len(inp)))
    return padded


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return (pred_flat == labels_flat).mean()


def create_dl(inputs, masks, labels, is_val: bool=True, bs=32) -> DataLoader:
    data = TensorDataset(inputs, masks, labels)
    sampler = SequentialSampler(data) if is_val else RandomSampler(data)
    return DataLoader(data, sampler=sampler, batch_size=bs)


def tokenise_texts(tokeniser, texts):
    out = []

    for text in texts:
        encoded = tokeniser.encode(*sent, add_special_tokens=True) if type(sent) == tuple \
            else tokeniser.encode(sent, add_special_tokens=True)
        out.append(encoded)
    return out


def calc_attention_mask(input_ids):
    attention_masks = []

    # For each sentence...
    for ids in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='Whether to train or inference in [\'train\', \'infer\']')
    parser.add_argument('--data_path', default='data/aclImdb/imdb', help='Path to data')
    parser.add_argument('--device', default='cuda:0', help='Device to run on')
    parser.add_argument('--seed', default=42, help='random seed, use -1 for non-determinism')
    parser.add_argument('--batch_size', default=16, help='batch size for train and inference')
    parser.add_argument('--lr', default=2e-5, help='initial learning rate of the AdamW optimiser')
    parser.add_argument('--opt_eps', default=1e-8, help='Epsilon for the AdamW parameter')
    parser.add_argument('--n_epochs', default=2, help='number of training epochs')
    parser.add_argument('--n_classes', default=2, help='number of target classes')
    parser.add_argument('--val_size', default=.1, help='relative size of the validation from the train set')
    parser.add_argument('--seq_len', default=256, help='The number of tokens to enter the model')

    opt_eps = 1e-8

    seed_everything(args.seed)
    
    df = pd.read_csv(args.data_path+"_train_clean.csv")
    df_train, df_validation = train_test_split(df, test_size=args.val_size, random_state=args.seed)
    df_test = pd.read_csv(args.data_path+"_test_clean.csv")

    # used for 2 text tasks like NLI
    if "content2" in df_train.columns:
        df_train["content"] = list(zip(df_train.content, df_train.content2))
        df_train = df_train.drop("content2", 1)
        df_validation["content"] = list(zip(df_validation.content, df_validation.content2))
        df_validation = df_validation.drop("content2", 1)
        df_test["content"] = list(zip(df_test.content, df_test.content2))
        df_test = df_test.drop("content2", 1)

    # Get the lists of sentences and their labels.
    sentences_train = df_train.content.values
    train_labels = df_train.label.values

    sentences_validation = df_validation.content.values
    val_labels = df_validation.label.values

    sentences_test = df_test.content.values
    test_labels = df_test.label.values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # this can be replaced with batch tokenisation in future release
    # Tokenize all sentences and map the tokens to their word IDs.
    train_inputs = tokenise_texts(tokenizer, sentences_train)
    val_inputs = tokenise_texts(tokenizer, sentences_validation)
    test_inputs = tokenise_texts(tokenizer, sentences_test)

    # Set the maximum sequence length.
    print('Padding/truncating all texts to %d values...' % args.seq_len)

    # Pad our input tokens with value 0.
    train_inputs = pad_sequences(train_inputs, maxlen=args.seq_len)
    val_inputs = pad_sequences(val_inputs, maxlen=args.seq_len)
    test_inputs = pad_sequences(test_inputs, maxlen=args.seq_len)

    # In newer version this is built into the tokeniser and can be replaced
    train_masks = calc_attention_mask(train_inputs)
    val_masks = calc_attention_mask(val_inputs)
    test_masks = calc_attention_mask(test_inputs)

    # Convert all inputs and labels into torch tensors
    train_inputs, val_inputs, test_inputs = torch.tensor(train_inputs), torch.tensor(val_inputs), torch.tensor(test_inputs)
    train_labels, val_labels, test_labels = torch.tensor(train_labels), torch.tensor(val_labels), torch.tensor(test_labels)
    train_masks, val_masks, test_masks = torch.tensor(train_masks), torch.tensor(val_masks), torch.tensor(test_masks)

    # Create the DataLoader for our training set.
    train_dataloader = create_dl(train_inputs, train_masks, train_labels, False, args.batch_size)
    val_dataloader = create_dl(val_inputs, val_masks, val_labels, True, args.batch_size)
    test_dataloader = create_dl(test_inputs, test_masks, test_labels, True, args.batch_size)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=args.n_classes,
                                                          output_attentions=False, output_hidden_states=False)
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.opt_eps)
    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # evaluation only - to make sure that accuracy is more or less random at the beginning
    print("Running Validation...")
    t0 = time.time()

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in val_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():        
            outputs = model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

    # training
    for epoch_i in range(0, args.n_epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.n_epochs))
        t0 = time.time()

        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 200 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(args.device)
            b_input_mask = batch[1].to(args.device)
            b_labels = batch[2].to(args.device)

            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)

        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        #               Validation
        print("Running Validation...")
        t0 = time.time()
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in val_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("  Accuracy: {0:.4f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("Training complete!")
    # torch.save(model.state_dict(), open(args.data_path+"_bert.pth", 'wb'))
