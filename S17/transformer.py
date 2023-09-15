# =============================================================================
# Transformer
# =============================================================================

from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import Counter
from os.path import exists
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import math
import re
from utils import *
from VIT_utils import *

class AttentionHead(nn.Module):
    """
    One head of the self-attention layer
    """

    def __init__(self, head_size, num_embed, block_size, dropout,**kwargs):
        super().__init__()
        
        self.head_size = head_size
        self.num_embed = num_embed
        self.block_size = block_size
        self.key = nn.Linear(self.num_embed, self.head_size, bias=False)
        self.query = nn.Linear(self.num_embed, self.head_size, bias=False)
        self.value = nn.Linear(self.num_embed, self.head_size, bias=False)
        # tril is a lower triangular matrix. it is not a parameter
        # of the model, so we assign it to the module using register_buffer
        self.register_buffer("tril", torch.tril(torch.ones(self.block_size, self.block_size)))

        # let's also add dropout
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        # Tril matrix (lower triagular matrix) is used to mask 
        # future positions (setting them to -inf) so that the
        # decoder "learns" to predict next words
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        # weighted aggregation of the values
        v = self.value(x)
        out = wei @ v  # (B,T,T) @ (B,T,C) ---> (B,T,C)
        return out
        
        
def attention(q, k, v, mask = None, dropout = None):
    scores = q.matmul(k.transpose(-2, -1))
    scores /= math.sqrt(q.shape[-1])

    #mask
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)

    scores = F.softmax(scores, dim = -1)
    scores = dropout(scores) if dropout is not None else scores
    output = scores.matmul(v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=None, out_dim=None, dropout=0.1, training_model ="BERT", 
                        num_heads=4, head_size= None, num_embed = 32, block_size = 8, dropout_gpt=None ,**kwargs):
        super().__init__()
        
        self.training_model = training_model
        
        if self.training_model == "GPT":
            
            self.num_heads = num_heads
            self.head_size=head_size
            self.num_embed=num_embed
            self.block_size=block_size
            self.dropout=dropout_gpt

            
            self.heads = nn.ModuleList(
            [
                AttentionHead(
                    head_size=self.head_size,
                    num_embed=self.num_embed,
                    block_size=self.block_size,
                    dropout=self.dropout,
                ) for _ in range(self.num_heads)
            ]
            )
            self.proj = nn.Linear(self.num_embed, self.num_embed)
        
        elif self.training_model == "BERT":
            
            self.n_heads = n_heads
            self.out_dim = out_dim

#           self.q_linear = nn.Linear(out_dim, out_dim)
#           self.k_linear = nn.Linear(out_dim, out_dim)
#           self.v_linear = nn.Linear(out_dim, out_dim)
            self.linear = nn.Linear(out_dim, out_dim*3)
            self.out_dim_per_head = self.out_dim // self.n_heads
            self.out = nn.Linear(self.out_dim, self.out_dim)
            self.dropout = nn.Dropout(dropout)
            
    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)

    def forward(self, x, y=None, mask=None):
        #in decoder, y comes from encoder. In encoder, y=x
        if self.training_model == "BERT":
            y = x if y is None else y
    
            qkv = self.linear(x) # BS * SEQ_LEN * (3*EMBED_SIZE_L)
            q = qkv[:, :, :self.out_dim] # BS * SEQ_LEN * EMBED_SIZE_L
            k = qkv[:, :, self.out_dim:self.out_dim*2] # BS * SEQ_LEN * EMBED_SIZE_L
            v = qkv[:, :, self.out_dim*2:] # BS * SEQ_LEN * EMBED_SIZE_L
    
            #break into n_heads
            q, k, v = [self.split_heads(t) for t in (q,k,v)]  # BS * SEQ_LEN * HEAD * EMBED_SIZE_P_HEAD
            q, k, v = [t.transpose(1,2) for t in (q,k,v)]  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
    
            #n_heads => attention => merge the heads => mix information
            scores = attention(q, k, v, mask, self.dropout) # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
            scores = scores.transpose(1,2).contiguous().view(scores.shape[0], -1, self.out_dim) # BS * SEQ_LEN * EMBED_SIZE_L
            out = self.out(scores)  # BS * SEQ_LEN * EMBED_SIZE
            
        elif self.training_model == "GPT":
            
            out = torch.cat([h(x) for h in self.heads], dim=-1)
        # apply the linear projection layer
            out = self.dropout(self.proj(out))
            
        return out

class FeedForward(nn.Module):
    def __init__(self, inp_dim=None, inner_dim=None, dropout=0.1, training_model = 'BERT',num_embed=32, dropout_gpt=None,**kwargs):
        super().__init__()
        self.training_model = training_model
        
        if self.training_model =='BERT':
            self.linear1 = nn.Linear(inp_dim, inner_dim)
            self.linear2 = nn.Linear(inner_dim, inp_dim)
            self.dropout = nn.Dropout(dropout)

        elif self.training_model =='GPT':
            self.num_embed=num_embed
            self.dropout=dropout_gpt
            
            self.net = nn.Sequential(
            # in the Attention is All You Need paper
            # authors are using the size of the ffwd layer 2048
            # and the output of the model is 512
            # so we apply the same factor of 4
            nn.Linear(self.num_embed, 4 * self.num_embed),
            nn.ReLU(),
            # apply the linear projection layer
            nn.Linear(4 * self.num_embed, self.num_embed),
            self.dropout,
        )
            
            
    def forward(self, x):
        if self.training_model =='BERT':
        #inp => inner => relu => dropout => inner => inp
            ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
            
        elif self.training_model =='GPT':
            ff =self.net(x)
            
        return ff
        
        
class TransformerBlock(nn.Module):
    """
    This class will group together MultiHead Attention and
    FeedForward NN, so that we can copy it in Transformer
    """

    def __init__(self, num_heads, block_size, num_embed, dropout_gpt, training_model, **kwargs):
        super().__init__()
        
        self.num_heads= num_heads
        self.block_size = block_size
        self.num_embed = num_embed
        self.dropout = dropout_gpt
        self.training_model = training_model
        
        self.head_size = num_embed // num_heads
        self.sa = MultiHeadAttention(
            num_heads=self.num_heads,
            head_size=self.head_size,
            num_embed=self.num_embed,
            block_size=self.block_size,
            dropout_gpt=self.dropout,
            training_model = self.training_model
        )
        self.ffwd = FeedForward(num_embed=self.num_embed, dropout_gpt=self.dropout,training_model = self.training_model)
        # add the layer normalization
        self.ln1 = nn.LayerNorm(self.num_embed)
        self.ln2 = nn.LayerNorm(self.num_embed)

    def forward(self, x):
        # "x +" is the skip (or residual) connection
        # it helps with optimization
        # also we apply layer normalization before self-attention
        # and feed-forward (a reshufle from original paper)
        
        # x = x + self.sa(self.ln1(x))
        # x = x + self.ffwd(self.ln2(x))
        
        x1 = self.ln1(x)
        x1 = self.sa(x1)
        x = x + x1
        x2 = self.ln2(x)
        x2 = self.ffwd(x2)
        x = x + x2
        
        return x

class EncoderLayer(nn.Module):
    def __init__(self, n_heads=None, inner_transformer_size=None, inner_ff_size=None, dropout=0.1,training_model=None ,**kwargs):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, inner_transformer_size, dropout,training_model)
        self.ff = FeedForward(inner_transformer_size, inner_ff_size, dropout,training_model)
        self.norm1 = nn.LayerNorm(inner_transformer_size)
        self.norm2 = nn.LayerNorm(inner_transformer_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x

class Transformer(nn.Module):
    def __init__(self, n_code=None,n_heads=4, embed_size = 32, inner_ff_size= None, n_embeddings=None, seq_len=None,dropout=.1,
                       vocab_size = 100, num_embed =32, block_size =8, num_heads=4, num_layers =4, dropout_gpt =0.2,
                       img_size:int=224, in_channels =3,patch_size:int=16, num_transformer_layers:int=12, embedding_dim:int=768,
                        mlp_size:int=3072, 
                 num_head:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000,
                         
                       training_model = 'BERT',**kwargs ):
        super().__init__()
        self.training_model = training_model
        
        if self.training_model == 'BERT':
            #n_code, n_heads, embed_size, inner_ff_size, len(dataset.vocab), seq_len, dropout
            
            self.n_code =n_code
            self.inner_ff_size =inner_ff_size
            self.embed_size = embed_size
            
            self.n_heads = n_heads
            self.seq_len = seq_len
            self.dropout = dropout
            self.n_embeddings = n_embeddings
    
            #model input
            self.embeddings = nn.Embedding(self.n_embeddings, self.embed_size)
            self.pe = PositionalEmbedding(self.embed_size, self.seq_len)
    
            #backbone
            encoders = []
            for i in range(n_code):
                encoders += [EncoderLayer(self.n_heads, self.embed_size, self.inner_ff_size, self.dropout,self.training_model)]
            self.encoders = nn.ModuleList(encoders)
    
            #language model
            self.norm = nn.LayerNorm(embed_size)
            self.linear = nn.Linear(embed_size, n_embeddings, bias=False)
            
        elif self.training_model == 'GPT':
            
            self.vocab_size = vocab_size
            self.num_embed = num_embed
            self.block_size = block_size
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.dropout = nn.Dropout(dropout_gpt)
            # each token reads the logits for the next token from a lookup table
            self.token_embedding_table = nn.Embedding(self.vocab_size, self.num_embed)
            # each position from 0 to block_size-1 will get its embedding
            self.position_embedding_table = nn.Embedding(self.block_size, self.num_embed)
            self.blocks = nn.Sequential(
                *[
                    TransformerBlock(
                        num_heads=self.num_heads,
                        block_size=self.block_size,
                        num_embed=self.num_embed,
                        dropout_gpt=self.dropout,
                        training_model = self.training_model
                    )
                    for _ in range(self.num_layers)
                ]
            )
            # we add the layer norm before the Linear layer
            self.ln_f = nn.LayerNorm(self.num_embed)
            self.lm_head = nn.Linear(self.num_embed, self.vocab_size)
            
        elif self.training_model == "VIT":
            # 3. Make the image size is divisble by the patch size 
            assert img_size % block_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {block_size}."
            
            # 4. Calculate number of patches (height * width/patch^2)
            self.num_patches = (img_size * img_size) // block_size**2
                    
            # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
            self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                                requires_grad=True)
            
            # 6. Create learnable position embedding
            self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                                requires_grad=True)
                    
            # 7. Create embedding dropout value
            self.embedding_dropout = nn.Dropout(p=dropout)
            
            # 8. Create patch embedding layer
            self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                                patch_size=block_size,
                                                embedding_dim=embedding_dim)
            
            # 9. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential()) 
            # Note: The "*" means "all"
            self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                                num_heads=num_head,
                                                                                mlp_size=mlp_size,
                                                                                mlp_dropout=mlp_dropout) for _ in range(num_layers)])
        
            # 10. Create classifier head
            self.classifier = nn.Sequential(
                nn.LayerNorm(normalized_shape=embedding_dim),
                nn.Linear(in_features=embedding_dim, 
                        out_features=num_classes)
            )


    def forward(self, x =None , targets=None):
        if self.training_model =='BERT':
            
            x = self.embeddings(x)
            x = x + self.pe(x)
            for encoder in self.encoders:
                x = encoder(x)
            x = self.norm(x)
            x = self.linear(x)
            
            return x
            
        elif self.training_model =='GPT':
            B, T = x.shape
            
            # idx and targets are (B,T) tensor of integers
            # the token_emb is (B, T, C), C = NUM_EMBED
            token_emb = self.token_embedding_table(x)
            # (T, C)
            posit_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
    
            x = token_emb + posit_emb
            # apply one head of self-attention
            x = self.blocks(x)
            # (B, T, vocab_size)
            logits = self.lm_head(x)
            # compute the loss
            if targets != None:
                # cross_entropy accepts inputs in a (batch_size, num_classes)
                # so we need to reformat our logits dimensions to
                # (batch_size * time, dim_vocabulary), time = block_size
                B, T, C = logits.shape
                logits = torch.reshape(logits, (B * T, C))
                targets = torch.reshape(targets, (B * T,))
                loss = F.cross_entropy(logits, targets)
            else:
                loss = None
            return logits, loss
            
        elif self.training_model =='VIT':
            # 12. Get batch size
            batch_size = x.shape[0]
            
            # 13. Create class token embedding and expand it to match the batch size (equation 1)
            class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

            # 14. Create patch embedding (equation 1)
            x = self.patch_embedding(x)

            # 15. Concat class embedding and patch embedding (equation 1)
            x = torch.cat((class_token, x), dim=1)

            # 16. Add position embedding to patch embedding (equation 1) 
            x = self.position_embedding + x

            # 17. Run embedding dropout (Appendix B.1)
            x = self.embedding_dropout(x)

            # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
            x = self.transformer_encoder(x)

            # 19. Put 0 index logit through classifier (equation 4)
            x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

            return x   

            

            
    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the context too the  last block_size tokens
            # because tokens don't communicate between blocks
            idx_crop = idx[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(idx_crop)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution with probabilities probs
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
            

# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:,:x.size(1)] #x.size(1) = seq_len
        
        
# 1. Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.
    
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """ 
    # 2. Initialize the class with appropriate variables
    def __init__(self, 
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()
        self.patch_size= patch_size
        
        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method 
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {patch_size}"
        
        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched) 
        # 6. Make sure the output shape has the right order 
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
        
        