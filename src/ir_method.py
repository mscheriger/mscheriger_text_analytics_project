import json
import logging
from elasticsearch import Elasticsearch, helpers
import requests
import pickle

logger = logging.getLogger(__name__)

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

def gendata(lines):
    #Helper function to load ARC corpus into elasticsearch database
    for line in lines:
        yield {
            "_index": "lines",
            "text": line,
        }

def get_maxscore(es,text):
    #Calculates max score from search query
    query = es.search(body={"query": {"match": {'text':text}}})
    return query['hits']['max_score']

def get_all_data(train=True):
    ##Loads multiple choice questions. train indicates loading training data, otherwise test
    if train:
        datasets = ('ARC-Challenge/ARC-Challenge-Dev.jsonl','ARC-Challenge/ARC-Challenge-Train.jsonl','ARC-Easy/ARC-Easy-Dev.jsonl','ARC-Easy/ARC-Easy-Train.jsonl')
    else:
        datasets = ('ARC-Challenge/ARC-Challenge-Test.jsonl','ARC-Easy/ARC-Easy-Test.jsonl')
    question_answers = []
    path = 'data/ARC-V1-Feb2018-2/'
    for dataset in datasets:
        for line in open(path + dataset, 'r',encoding="utf8"):
            qa = json.loads(line)
            if dataset in ('ARC-Challenge/ARC-Challenge-Dev.jsonl','ARC-Challenge/ARC-Challenge-Train.jsonl','ARC-Challenge/ARC-Challenge-Test.jsonl'):
                qa['type'] = 'challenge'
            else:
                qa['type'] = 'easy'
            question_answers.append(qa)
    return question_answers

def get_answer_IR(es,question_answer=None,question=None,choices=None):
    #Makes prediction - whichever question-answer combo returns highest score
    max_score = 0
    answer = ''
    if question_answer is None:
        q = question
        choice_list = []
        for choice,label in zip(choices,['A','B','C','D']):
            choice_list.append({'text':choice,'label':label})
        choices = choice_list
    else:
        q = question_answer['question']['stem']
    for choice in choices:
        q_a = ' '.join([q,choice['text']])
        new_score = get_maxscore(es,q_a)
        if new_score > max_score:
            answer = choice['label']
            max_score = new_score
    return answer

def calculate_accuracy(question_answers):
    #Calculates accuracy - question answers must have a prediction key
    correct = 0
    for qa in question_answers:
        if qa['answerKey'] == qa['prediction']:
            correct += 1
    return correct/len(question_answers)

def build_elastic_corpus():
    lines = open('./data/ARC-V1-Feb2018-2/ARC_Corpus.txt',encoding="utf8").readlines()
    logger.info('Loaded ARC Corpus')
    print('Loaded ARC Corpus')
    helpers.bulk(es,gendata(lines))
    print('database built')
    pass

def run_IR(train=True):
    #Runs IR predictions. Returns accuracy metric
    question_answers = get_all_data(train)
    print('Loaded questions')

    path = 'obj/prediction_IR'
    if train:
        path += '_train.pkl'
    else:
        path += '_test.pkl'

    for qa in question_answers:
        qa['prediction'] = get_answer_IR(es,qa)
    with open(path, 'wb') as f:
        pickle.dump(question_answers, f)
    return calculate_accuracy(question_answers)
