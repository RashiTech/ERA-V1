import torch
from pytorch_lightning import LightningModule
from train import *
from dataset import BilingualDataset
from model import *
from torch import optim
from torch.utils.data import DataLoader
import config

class transformer_PL(LightningModule):
    def __init__(self):
        super(transformer_PL, self).__init__()
        
        self.config = get_config()
        
        
        self.batch_size =self.config['batch_size']
        self.learning_rate = self.config['lr']
        self.num_epochs = self.config['num_epochs']
        

    def forward(self, x):
        return self.model(x)
        
  
    def training_step(self, batch, batch_idx):
        self.encoder_input = batch['encoder_input'] # (b, seq_len) 
        self.decoder_input = batch['decoder_input']# (B, seq_len) 
        self.encoder_mask = batch['encoder_mask']# (8, 1, 1, seq_len)
        self.decoder_mask = batch['decoder_mask']# (8, 1, seq_len, seq_len)
        self.label = batch['label']
        
        self.encoder_output= self.model.encode(self.encoder_input, self.encoder_mask) # (B, seq_len, d_model) 
        self.decoder_output = self.model.decode(self.encoder_output, self.encoder_mask, self.decoder_input,self.decoder_mask) # 
        self.proj_output = self.model.project(self.decoder_output)
        loss = self.criterion(self.proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), self.label.view(-1))
        self.log(f"train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        
        # Backpropagate the loss 
        loss.backward(retain_graph=True) 
            
         
        return loss

    
    def validation_step(self, batch, batch_idx):
        self.encoder_input = batch['encoder_input'] # (b, seq_len) 
        self.decoder_input = batch['decoder_input']# (B, seq_len) 
        self.encoder_mask = batch['encoder_mask']# (8, 1, 1, seq_len)
        self.decoder_mask = batch['decoder_mask']# (8, 1, seq_len, seq_len)
        self.label = batch['label']
        
        self.encoder_output= self.model.encode(self.encoder_input, self.encoder_mask) # (B, seq_len, d_model) 
        self.decoder_output = self.model.decode(self.encoder_output, self.encoder_mask, self.decoder_input,self.decoder_mask) # 
        self.proj_output = self.model.project(self.decoder_output) # (B, seq_len, vocab_size)
        self.val_loss = self.criterion(self.proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), self.label.view(-1))
        self.log(f"val_loss", self.val_loss, on_epoch=True, prog_bar=True, logger=True)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], eps=1e-9 )
        return optimizer
        
    def prepare_data(self):  
        
        src_lng =  self.config["lang_src"]
        tgt_lang = self.config["lang_tgt"]
        # download
        # It only has the train split, so we divide it overselves 
        self.ds_raw = load_dataset('opus_books', f"{src_lng}-{tgt_lang}", split='train')        
        
       
        
    def setup(self, stage=None):
         # Build tokenizers 
        self.tokenizer_src = get_or_build_tokenizer(self.config, self.ds_raw, self.config['lang_src'])
        self.tokenizer_tgt = get_or_build_tokenizer(self.config, self.ds_raw, self.config['lang_tgt'])
        
        # Keep 90% for training, 10% for validation 
        train_ds_size = int(0.9* len(self.ds_raw))
        val_ds_size = len(self.ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(self.ds_raw, [train_ds_size, val_ds_size])

        self.train_ds = BilingualDataset(train_ds_raw, self.tokenizer_src, self.tokenizer_tgt, self.config['lang_src'], 
                                    self.config['lang_tgt'], self.config['seq_len'])
        self.val_ds = BilingualDataset(val_ds_raw, self.tokenizer_src, self.tokenizer_tgt, self.config['lang_src'], 
                                    self.config['lang_tgt'], self.config['seq_len'])

        # Find the maximum length of each sentence in the source and target sentence 
        max_len_src = 0 
        max_len_tgt = 0 

        for item in self.ds_raw: 
            src_ids = self.tokenizer_src.encode(item['translation'][self.config['lang_src']]).ids
            tgt_ids = self.tokenizer_src.encode(item['translation'][self.config['lang_tgt']]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f'Max length of source sentence: {max_len_src}') 
        print(f'Max length of target sentence: {max_len_tgt}') 
        
        self.model = get_model(self.config, self.tokenizer_src.get_vocab_size(), self.tokenizer_tgt.get_vocab_size())
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)
        
        

    def train_dataloader(self):
            return DataLoader(self.train_ds, batch_size=self.config['batch_size'], shuffle=True,  num_workers=4, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
            return DataLoader(self.val_ds, batch_size=1, shuffle=False,  num_workers=4, persistent_workers=True, pin_memory=True) 