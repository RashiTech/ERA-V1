from model import build_transformer
from dataset import BilingualDataset, causal_mask 
from config import get_config, get_weights_file_path

#import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split 
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

#Huggingface datasets and tokenizers 
from datasets import load_dataset 
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer 
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics

from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device): 
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as (source).to(device) 
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask (decoder_input.size(1)).type_as (source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project (out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim = 1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step,writer, num_examples=2): 
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted = []
    try:
    # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split() 
            console_width = int(console_width)
    except:
    # If we can't get the console width, use 80 as default 
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch[ "encoder_input"].to(device) # (b, seq_len) 
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
                                      
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12} {source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
                
    if writer:
        # Evaluate the character error rate
        #Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric (predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate() 
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step) 
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_all_sentences (ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences (ds, lang), trainer=trainer) 
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_ds (config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src']) 
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation 
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len (ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

                                                                                                   
    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    # for item in ds_raw:
    #     src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
    #     tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids 
    #     max_len_src = max(max_len_src, len(src_ids)) 
    #     max_len_tgt = max(max_len_tgt, len(tgt_ids))

    # print(f'Max length of source sentence: {max_len_src}') 
    # print(f'Max length of target sentence: {max_len_tgt}')
    
    filtered_train_ds = [k for k in train_ds_raw if len(k["translation"][config['lang_src']])  < 150]
    filtered_train_ds = [k for k in filtered_train_ds if len(k["translation"][config['lang_tgt']])  < 150]
    filtered_train_ds = [k for k in filtered_train_ds if len(k["translation"][config['lang_tgt']]) - len(k["translation"][config['lang_src']])  < 10]
    
    for item in filtered_train_ds:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids 
        max_len_src = max(max_len_src, len(src_ids)) 
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of (filtered) source sentence: {max_len_src}') 
    print(f'Max length of (filtered) target sentence: {max_len_tgt}')
    

    train_ds = BilingualDataset(filtered_train_ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
     
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn = collate_fn) 
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def collate_fn(batch):

    encoder_input_max = max(x["encoder_str_length"] for x in batch)
    decoder_input_max = max(x["decoder_str_length"] for x in batch)
    
    max_len = max(encoder_input_max , decoder_input_max)

    encoder_inputs = []
    decoder_inputs = []
    encoder_mask = []
    decoder_mask = []
    src_text =[]
    tgt_text= []
    labels = []
    enc_num_padding_tokens = 0
    dec_num_padding_tokens = 0
    
    
    for b in batch:
    
        enc_input_tokens = b['encoder_input']
        dec_input_tokens  = b['decoder_input']
    
        # Add sos, eos and padding to each sentence
        if len(enc_input_tokens) > max_len - 2:
            enc_input_tokens = enc_input_tokens[:max_len - 2]
            
        enc_num_padding_tokens = max_len - len(enc_input_tokens) - 2 # We will add <s> and </s> 
        # We will only add <s>, and </s> only on the label
        if len(dec_input_tokens) > max_len - 1:
            dec_input_tokens = dec_input_tokens[:max_len - 1]
        dec_num_padding_tokens = max_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        
        #Ad <s> and </s> token 
        encoder_input = torch.cat(
            [
                b['sos_token'],
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                b['eos_token'],
                torch.tensor([b['pad_token']] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                b['sos_token'],
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([b['pad_token']] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                b['eos_token'],
                torch.tensor([b['pad_token']] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        
        encoder_mask1 =  (encoder_input != b['pad_token']).unsqueeze(0).unsqueeze(0).int() # (1, 1, 
        decoder_mask1 = (decoder_input != b['pad_token']).unsqueeze(0).int() & causal_mask(decoder_input.size(0))
        
       # Double check the size of the tensors to make sure they are all seq_len long 
        # assert encoder_input.size(0) == max_len
        # assert decoder_input.size(0) == max_len 
        # assert label.size(0) == max_len

        encoder_inputs.append(encoder_input)
        decoder_inputs.append(decoder_input)
        encoder_mask.append((encoder_mask1[0,0,:max_len]).unsqueeze(0).unsqueeze(0).unsqueeze(0).int())
        decoder_mask.append((decoder_mask1[0,:max_len,:max_len]).unsqueeze(0).unsqueeze(0))
        labels.append(label)
        src_text.append(b['src_text'])
        tgt_text.append(b['tgt_text'])
    
    return {
        "encoder_input":torch.vstack(encoder_inputs),
        "decoder_input":torch.vstack(decoder_inputs),
        "encoder_mask":torch.vstack(encoder_mask),      
        "decoder_mask":torch.vstack(decoder_mask),
        "label":torch.vstack(labels),
        "src_text":src_text,
        "tgt_text":tgt_text
      }
    



def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device) 
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    #scheduler - OneCycle LR
    MAX_LR = 10**-3
    EPOCHS = 25
    STEPS_PER_EPOCH = len(train_dataloader)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr= MAX_LR,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        pct_start=1/10,
        div_factor=10,
        three_phase=True,
        final_div_factor=10,
        anneal_strategy='linear'
    )
    
    

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print (f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        print("preloaded")

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len) 
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len) 
            encoder_mask = batch['encoder_mask'].to(device) # (8, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (8, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output= model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model) 
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # 
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)) 
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step) 
            writer.flush()

            # Backpropagate the loss
            loss.backward() 

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1

        # Run validation at the end of every epoch
        #run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
                       
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}") 
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)