
import whisperx
import gc
import torch
import pydub
import ffmpeg
from pydub import AudioSegment
from transformers import Autotokenizer

device = "cuda" if torch.cuda.is_available() else 'cpu'


def projection_audio(audio_file):
    '''
    converts audio to transcript to tokens to embeddings to be further inputted to phi model
    '''

    # Transcription
    batch_size = 1 
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    song = AudioSegment.from_mp3( audio_file )
    extract = song[:10*1000] # accepting only initial 10 secs speech
    extract.export( 'audio.mp3', format="mp3")

    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    audio_file = '/content/audio.mp3'

    audio = whisperx.load_audio(audio_file)

    result = model.transcribe(audio, batch_size=batch_size)

    transcript =  result["segments"][0]['text']

    #Tokenization and generating embeddings for making input ready to phi-2
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    
    aud_tokens = tokenizer(transcript, return_tensors="pt", return_attention_mask=False)
    audio_token_embeds = self.phi_model.model.embed_tokens(aud_tokens['input_ids'].squeeze(0).to(device))
    
    return audio_token_embeds
    


