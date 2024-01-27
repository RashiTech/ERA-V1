
import whisperx
import gc
import torch
import pydub
import ffmpeg
from pydub import AudioSegment

device = "cuda" if torch.cuda.is_available() else 'cpu'

def speech2text(audio_file):

    batch_size = 1 
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    song = AudioSegment.from_mp3( audio_file )
    extract = song[:10*1000] # accepting only initial 10 secs speech
    extract.export( 'audio.mp3', format="mp3")

    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    audio_file = '/content/audio.mp3'

    audio = whisperx.load_audio(audio_file)

    result = model.transcribe(audio, batch_size=batch_size)

    return result["segments"][0]['text']

