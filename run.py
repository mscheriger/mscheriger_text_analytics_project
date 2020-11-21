from src.ir_method import run_IR, build_elastic_corpus
from src.bert_method import run_bert, get_relevant_text
from src.prediction_results import get_predictions 
import pickle
import yaml
import logging
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Create elasticsearch database or run models")
parser.add_argument('-c', '--create_corpus', action='store_true', default=False, help='Create elasticsearch database')
parser.add_argument('-g', '--get_text', action='store_true', default=False, help='Gets context for all questions using elasticsearch')
parser.add_argument('-i', '--ir', action='store_true', default=False, help='Runs IR model for all questions and answers using elasticsearch')
parser.add_argument('-b', '--bert', action='store_true', default=False, help='Runs BERT model')
parser.add_argument('-p', '--predictions', action='store_true', default=False, help='Calculate prediction accuracy for easy and hard questions for all models')
args = parser.parse_args()

try: 
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
except FileNotFoundError as e:
    logger.error('Cannot find config file')
    exit()

lines = config['lines']
use_rouge = config['use_rouge'] 
train = config['train']
ten = config['ten']

##Create the Elasticsearch database
if args.create_corpus:
    build_elastic_corpus()

##Gets the context for each question
if args.get_text:
    get_relevant_text(lines=lines)

##Runs the IR method for all multiple choice questions
if args.ir:
    answer = run_IR(train)

##Runs the BERT method for all multiple choice questions
if args.bert:
    answer = run_bert(train,use_rouge,ten)

##Prints the results from all models
if args.predictions:
    get_predictions()
