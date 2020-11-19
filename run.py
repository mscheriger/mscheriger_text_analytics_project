from src.ir_method import run_IR, build_elastic_corpus
from src.bert_method import run_bert, get_relevant_text
import pickle

build_corpus = False 
get_text = False 
runIR = False
runBert = True

use_rouge = True
train=True
ten=False

if build_corpus:
    build_elastic_corpus()

if get_text:
    get_relevant_text(lines=10)

if runIR:
    answer = run_IR(False)
    print(answer)

if runBert:
    answer = run_bert(train,use_rouge,ten)
    print(answer)
