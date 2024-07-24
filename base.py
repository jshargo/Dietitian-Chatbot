import transformers
import accelerate
import peft
from datasets import load_dataset

ds = load_dataset("scene_parse_150", split="train[:150]")

ds = ds.train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]