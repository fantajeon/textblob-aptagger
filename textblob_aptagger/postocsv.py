import pickle
import numpy
import pdb
import random
from load_trainset_8K import Tokenizer
import os


#load_class = False
load_class = True
class_pick = "./csv/classes.pkl"

if not load_class:
    data_root = "/mywork/8K/usa_stock_parsing/8K_POS/"
else:
    data_root = "/mywork/8K/usa_stock_parsing/8K_POS_test/"
csv_root = "./csv/"
def ndprint(a, format_string ='{0:.2f}'):
    print ([format_string.format(v,i) for i,v in enumerate(a)])

def make_directories():
    if not os.path.exists(csv_root):
        os.mkdir( csv_root )

def convert_to_csv():
    if load_class:
        f = open(class_pick,"rb")
        tag_dict = pickle.load(f)
        f.close()
    else:
        tag_dict = dict()
        tag_dict["UNK"] = 0

    make_directories()
    if not load_class:
        filelist = "/mywork/8K/usa_stock_parsing/8K_POS/8K_poslist.txt"
    else:
        filelist = "/mywork/8K/usa_stock_parsing/8K_POS_test/8K_test_poslist.txt"
    fp = open(filelist, "rt")
    trainlist = [name.strip() for name in fp.readlines()]
    fp.close()

    if not load_class:
        csv_filename = os.path.join( csv_root, "train.csv" )
    else:
        csv_filename = os.path.join( csv_root, "test.csv" )


    fp_train = open(csv_filename, "wt")
    for filename in trainlist:
        taglist = []
        wordlist = []
        filename = os.path.join(data_root, filename.strip())
        tokenizer = Tokenizer()
        if not tokenizer.open(filename):
            continue
        stok = tokenizer.readTokens()

        for words, tags in stok:
            wordlist += words
            taglist += tags
            wordlist += ['\n']
            taglist += ['UNK']

            for t in tags:
                if not t in tag_dict:
                    tag_dict[t] = len(tag_dict)

        # tag중심으로, sentence를 앞위 495단어씩을 뗀다.
        for i in range(len(wordlist)):
            t = taglist[i]
            if not t == "UNK" or random.randint(0, 10) == 0:
                buf = numpy.zeros((1011,),dtype=str)
                lidx = 489
                cidx = 490
                ridx = 521

                sidx = i-1
                try:
                    while True:
                        for w in reversed(wordlist[sidx]):
                            buf[lidx] = w
                            lidx -= 1
                            if lidx <0:
                                raise IndexError()
                        sidx -= 1
                        lidx -= 1
                        if lidx <0:
                            raise IndexError()
                except IndexError as e:
                    pass

                try:
                    for w in wordlist[i]:
                        buf[cidx] = w
                        cidx+=1
                        if cidx >= 512:
                            raise IndexError()
                except IndexError as e:
                    pass

                sidx = i+1
                try:
                    while True:
                        for w in wordlist[sidx]:
                            buf[cidx] = w
                            cidx += 1
                            if cidx >= 1011:
                                raise IndexError()
                        cidx += 1
                        if cidx >= 1011:
                            raise IndexError()
                        sidx += 1
                except IndexError as e:
                    pass
                tok_id = tag_dict[t]
                str_buf = ''
                for c in buf.tolist():
                    if c == '':
                        str_buf += ' '
                    elif c == '\"':
                        str_buf += '\"\"'
                    elif c == '\r':
                        continue
                    elif c == '\n':
                        str_buf += '\\\\n'
                    else:
                        str_buf += c

                keyword = wordlist[i]
                keyword = keyword.replace("\n", "\\\\n")
                keyword = keyword.replace("\"", "\"\"")
                line = "{},\"{}\",\"{}\"\n".format( tok_id + 1, keyword, str_buf)
                fp_train.write(line)

    fp_train.close()

    if not load_class:
        f = open(class_pick, "wb")
        pickle.dump(tag_dict, f)
        f.close()

if __name__ == "__main__":
	convert_to_csv()
