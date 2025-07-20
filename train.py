from pathlib import Path
import modal
from torch.utils.data import Dataset
import pandas as pd

app = modal.App(name="audio-cnn")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
         .run_commands([
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ])
         .add_local_python_source("model"))

volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
modal_volume = modal.Volume.from_name("esc-model", create_if_missing=True)

class ESC50Dataset(Dataset):
     def __init__(self, data_dir, metadata_file, split="train", transform=None):
         super().__init__()
         self.data_dir = Path(data_dir)
         self.metadata = pd.read_csv(metadata_file)
         self.split = split
         self.transform = transform

@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": modal_volume}, timeout=60*60*3)
def train():
    print("training")


@app.local_entrypoint()
def main():
    train.remote()