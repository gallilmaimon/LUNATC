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


if __name__ == '__main__':
    seed_val = 42
    data_path = "data/mnli2/mnli"
    val_size = 0.1
    MAX_LEN = 256  # this was chosen based n the data distribution so most texts aren't truncated
    batch_size = 16
    NUM_LABELS = 3  # 3 for MNLI, 2 for IMDB, Toxic-WIKI, PUBMED
    epochs = 4  # 2 for text classification, 4 for MNLI
    device = "cuda"

    seed_everything(seed_val)
    
    df = pd.read_csv(data_path+"_train_clean.csv")
    df_train, df_validation = train_test_split(df, test_size=val_size, random_state=seed_val)
    df_test = pd.read_csv(data_path+"_test_clean.csv")
    
    df_train = df_train.head(1000)
    df_validation = df_validation.head(1000)
    df_test = df_test.head(1000)
    
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
    labels_train = df_train.label.values

    sentences_validation = df_validation.content.values
    labels_validation = df_validation.label.values

    sentences_test = df_test.content.values
    labels_test = df_test.label.values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # TODO: remove duplicate code to function, maybe use batch tokenisation - 0
    # Tokenize all sentences and map the tokens to their word IDs.
    input_ids_train = []

    # For every text in train
    for sent in sentences_train:
        encoded_sent = tokenizer.encode(
            *sent,                      # Sentence to encode.
            add_special_tokens=True,    # Add '[CLS]' and '[SEP]'
        )

        input_ids_train.append(encoded_sent)
    print("Train done!")

    # For every text in validation
    input_ids_validation = []
    for sent in sentences_validation:
        encoded_sent = tokenizer.encode(
            *sent,                      # Sentence to encode.
            add_special_tokens=True,    # Add '[CLS]' and '[SEP]'
        )

        input_ids_validation.append(encoded_sent)
    print("Validation done!")

    # For every text in test
    input_ids_test = []
    for sent in sentences_test:
        encoded_sent = tokenizer.encode(
            *sent,                      # Sentence to encode.
            add_special_tokens=True,    # Add '[CLS]' and '[SEP]'
        )

        input_ids_test.append(encoded_sent)
    print("Test done!")

    # Set the maximum sequence length.
    print('Padding/truncating all texts to %d values...' % MAX_LEN)

    # Pad our input tokens with value 0.
    input_ids_train = pad_sequences(input_ids_train, maxlen=MAX_LEN)
    input_ids_validation = pad_sequences(input_ids_validation, maxlen=MAX_LEN)
    input_ids_test = pad_sequences(input_ids_test, maxlen=MAX_LEN)

    # TODO: remove duplicate code and over documentation, maybe use proper tokenisation instead of this - 1
    attention_masks_train = []

    # For each sentence...
    for sent in input_ids_train:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks_train.append(att_mask)

    # Create attention masks
    attention_masks_validation = []

    # For each sentence...
    for sent in input_ids_validation:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks_validation.append(att_mask)

    attention_masks_test = []

    # For each sentence...
    for sent in input_ids_test:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks_test.append(att_mask)

    # TODO: maybe this can be avoided - 2
    # rename
    train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = input_ids_train, input_ids_validation, input_ids_test, labels_train, labels_validation, labels_test
    train_masks, val_masks, test_masks = attention_masks_train, attention_masks_validation, attention_masks_test

    # Convert all inputs and labels into torch tensors
    train_inputs, val_inputs, test_inputs = torch.tensor(train_inputs), torch.tensor(val_inputs), torch.tensor(test_inputs)
    train_labels, val_labels, test_labels = torch.tensor(train_labels), torch.tensor(val_labels), torch.tensor(test_labels)
    train_masks, val_masks, test_masks = torch.tensor(train_masks), torch.tensor(val_masks), torch.tensor(test_masks)

    # Create the DataLoader for our training set.
    train_dataloader = create_dl(train_inputs, train_masks, train_labels, False, batch_size)
    val_dataloader = create_dl(val_inputs, val_masks, val_labels, True, batch_size)
    test_dataloader = create_dl(test_inputs, test_masks, test_labels, True, batch_size)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS,
                                                          output_attentions=False, output_hidden_states=False)
    model.cuda()

    # TODO: move magic numbers
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # evaluation only - to make sure that accuracy is more or less random at the beginnig
    print("")
    print("Running Validation...")
    device = "cuda"
    t0 = time.time()

    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in val_dataloader:

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():        
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

    
    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 200 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

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
            batch = tuple(t.to(device) for t in batch)
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
    # torch.save(model.state_dict(), open(data_path+"_bert.pth", 'wb'))
