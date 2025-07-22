from llava.train.multi_sketch.train_garmentcode_outfit_sitian import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
    # train()
