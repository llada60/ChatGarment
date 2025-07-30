import sys
sys.path.append('/home/ids/liliu/projects/ChatGarment')
from llava.train.multi_sketch.train_garmentcode_outfit_sitian import train
import os
os.environ["MASTER_PORT"] = "10503"
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
    # train()
