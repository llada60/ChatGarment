from llava.train.image.train_garmentcode_outfit import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
