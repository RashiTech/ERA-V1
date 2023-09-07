from pathlib import Path

def get_config():
    return {
        #"batch_size": 2048,
        "batch_size": 16,
        "num_epochs": 24, 
        "lr": 10**-4, 
        "seq_len": 160,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": True,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
        }

def get_weights_file_path(config, epoch: str): 
    model_folder = config["model_folder"] 
    model_basename = config["model_basename"] 
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)