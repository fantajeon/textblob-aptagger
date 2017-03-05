import pudb
import pdb
from load_trainset_8K import Tokenizer
from taggers import PerceptronTagger
import os

model_root = "/data/8K/usa_stock_parsing/textblob-aptagger/textblob_aptagger/saved"
#data_root = "/data/8K/usa_stock_parsing/8K_POS/"
data_root = "/data/8K/usa_stock_parsing/8K_POS_test/"
out_dir = "/data/8K/usa_stock_parsing/8K_POS_out/"

def predict(tagger, filename):

    sentences = []
    filename = os.path.join(data_root, filename)
    tokenizer = Tokenizer()
    if not tokenizer.open(filename):
        return
    s1 = tokenizer.readTokens()
    for line in s1:
        sentences += [line[0]]

    ptags = tagger.tag( sentences )

    sentences = []
    for idx, t in enumerate(ptags):
        if t[1] == "UNK":
            sentences.append( t[0] )
        elif t[1] == "END":
            sentences.append( "\r\n") 
        else:
            sentences.append( "{}/{}".format(t[0], t[1]) )

    basename = os.path.basename(filename)
    output_filename = os.path.join(out_dir, "{}_tagged.txt".format(basename))
    with open(output_filename, "wt") as f:
        f.write( " ".join(sentences) )
        print ("output->{}".format(output_filename))

def run_test_model():
    filelist = "/data/8K/usa_stock_parsing/8K_POS_test/8K_test_poslist.txt"
    fp = open(filelist, "rt")
    trainlist = [name.strip() for name in fp.readlines()]
    fp.close()

    #model_name = os.path.join( model_root, "model__121.pkl" )
    #model_name = os.path.join( model_root, "model__1759.pkl" )
    model_name = os.path.join( model_root, "model__best.pkl" )
    tagger = PerceptronTagger(load=False)
    tagger.load(model_name)

    for filename in trainlist:
        print ("predict file:{}".format(filename))
        predict(tagger,filename)
        return

if __name__ == "__main__":
    run_test_model()
