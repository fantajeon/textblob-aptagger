import random
from load_trainset_8K import Tokenizer
from taggers import PerceptronTagger
import os


data_root = "/data/8K/usa_stock_parsing/8K_POS/"
model_root = "./saved/"

def make_directories():
    if not os.path.exists(model_root):
        os.mkdir( model_root )

def run_train_model():
    make_directories()
    filelist = "/data/8K/usa_stock_parsing/8K_POS/8K_poslist.txt"
    fp = open(filelist, "rt")
    trainlist = [name.strip() for name in fp.readlines()]
    fp.close()

    tagger = PerceptronTagger(load=False)
    loop = 0

    sentences = []
    for filename in trainlist:
        filename = os.path.join(data_root, filename.strip())
        tokenizer = Tokenizer()
        if not tokenizer.open(filename):
            continue
        sentences += tokenizer.readTokens()


    model_name = os.path.join( model_root, "model_" )
    tagger.train(sentences, model_name, nr_iter=2000000)

    loop += 1

if __name__ == "__main__":
    run_train_model()
