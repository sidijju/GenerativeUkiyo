from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir="data/train")
dataset.push_to_hub("sidijju/japanese-woodblock-prints")