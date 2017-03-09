import re
import pdb

class Tokenizer():
    def __init__(self):
        self.spliter = re.compile("^(.+)/(\[fc:.+\])(.*)", re.IGNORECASE)

    def open(self, filename):
        f = open(filename, "rt")
        if f is None:
            return False
        self.fp = f
        return True

    def close(self):
        f.close()

    def stringToToken(self, txt):
        if len(txt) == 0:
            return None
        m = self.spliter.match(txt)
        if m is None:
            return (txt, "UNK")
        else:
            return (m.group(1)+m.group(3), m.group(2))

    def normalize(self, txt):
        pat_list = [
                ('!YRS', re.compile('\d{2,4}[/\-~]\d{1,2}', re.IGNORECASE)),
                ('!YRS', re.compile('\d{2,4}[/\-~]\d{1,2}[/\-~]\d{1,2}', re.IGNORECASE)),
                ('!#DT', re.compile('\$?\s*\(?[\d,.]+\)?', re.IGNORECASE)), 
                ('!#DT', re.compile('\$?\(?[\d,.]+\)?', re.IGNORECASE))
                ]
        for norm_txt, matcher in pat_list:
            m = matcher.match(txt[0])
            if not m is None:
                return (norm_txt, txt[1])
        return txt


    def readTokens(self):
        sentences = []
        for line in self.fp.readlines():
            raw_word = line.split(" ")
            toks = []
            words = []
            for w in raw_word:
                tok = self.stringToToken(w.strip())
                if tok is None:
                    continue
                ntok = self.normalize(tok)
                words.append(ntok[0])
                toks.append(ntok[1])
            sentences.append((words,toks))
        return sentences


filename = "/data/8K/usa_stock_parsing/8K_POS/_1001233_urn_tag_sec.gov-2008_accession-number=0001144204-16-129622.html.pos"
if __name__ == "__main__":
    tokenizer = Tokenizer()
    if not tokenizer.open(filename):
        print ("error")
    toks = tokenizer.readTokens()
    print (toks)
