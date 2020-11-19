import torch
import pickle
import rouge
from elasticsearch import Elasticsearch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from nltk .tokenize import word_tokenize
from nltk.corpus import stopwords
from ir_method import calculate_accuracy, get_all_data

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
sw = stopwords.words('english')  

def get_answer(line):
    question = line['question']['stem']
    context = line['context']

    input_ids = tokenizer.encode(question,context)
    if len(input_ids)>512:
        input_ids = input_ids[:511]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    sep_index = input_ids.index(tokenizer.sep_token_id)

    num_question = sep_index + 1
    num_context = len(input_ids) - num_question
    segment_ids = [0]*num_question + [1]*num_context

    start_scores, end_scores = model(torch.tensor([input_ids]),
                                 token_type_ids=torch.tensor([segment_ids]))

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    answer = ' '.join(tokens[answer_start:answer_end+1])
    return answer

def get_answer_BERT(es,question,choices):
    context = get_paragraph(es,question)
    line = {'question':{'stem':question},'context':context}
    choice_list = []
    for label, choice in zip(['A','B','C','D'],choices):
        choice_list.append({'label':label,'text':choice})
    line['question']['choices'] = choice_list
    answer = get_answer(line)
    return make_prediction(line,answer)

def similarity(text1,text2):
    list1 = word_tokenize(text1)  
    list2 = word_tokenize(text2) 
    l1 =[];l2 =[] 
  
    set1 = {w for w in list1 if not w in sw}  
    set2 = {w for w in list2 if not w in sw} 
  
    union = set1.union(set2)  
    for w in union: 
        if w in set1: l1.append(1)
        else: l1.append(0) 
        if w in set2: l2.append(1) 
        else: l2.append(0) 
    c = 0
  
    # cosine formula  
    for i in range(len(union)): 
        c+= l1[i]*l2[i] 
    try:
        cosine = c / float((sum(l1)*sum(l2))**0.5) 
    except ZeroDivisionError:
        cosine = 0
    return cosine

def rouge_similarity(text1,text2):
    evaluator = rouge.Rouge()

    scores = evaluator.get_scores(text1,text2)
    return scores['rouge-l']['f']

def make_prediction(line,answer,use_rouge=False):
    prediction='A'
    score=0
    for choice in line['question']['choices']:
        if use_rouge:
            new = rouge_similarity(choice['text'],answer)
        else:
            new = similarity(choice['text'],answer)
        if new>score:
            prediction = choice['label']
            score = new
    return prediction

def run_bert(train=True,use_rouge=False,ten=False):
    ##Runs Bert predictions. Returns accuracy
    path = 'obj/prediction_bert'
    if train:
        if ten:
            path += '_train_10.pkl'
            data = pickle.load(open( "obj/questions_with_context_train_10.pkl", "rb" ))
        else:
            path += '_train.pkl'
            data = pickle.load(open( "obj/questions_with_context_train.pkl", "rb" ))
    else:
        if ten:
            path += '_test_10.pkl'
            data = pickle.load(open( "obj/questions_with_context_test_10.pkl", "rb" ))
        else:
            path += '_test.pkl'
            data = pickle.load(open( "obj/questions_with_context_test.pkl", "rb" ))

    print('Getting answers')
    answers = list(map(get_answer,data))
    print('Making predictions')
    for qa,answer in zip(data,answers):
        qa['prediction'] = make_prediction(qa,answer,use_rouge)

    if use_rouge:
        path = path[:-4]
        path += '_rouge.pkl'

    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return calculate_accuracy(data)

def get_paragraph(es,text,lines=5):
    #Returns top search results as a paragraph
    query = es.search(body={"query": {"match": {'text':text}}})
    return ' '.join([h['_source']['text'] for h in query['hits']['hits'][:lines]])

def get_relevant_text(lines=5):
    question_answers_train = get_all_data()
    print('Loaded Training questions')
    question_answers_test = get_all_data(train=False)
    print('Loaded Test questions')

    for qa in question_answers_train:
        qa['context'] = get_paragraph(es,qa['question']['stem'],lines)

    path = 'obj/questions_with_context_train'
    if lines==10:
        path += '_10'
    path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(question_answers_train, f)

    print('saved train data')
    for qa in question_answers_test:
        qa['context'] = get_paragraph(es,qa['question']['stem'],lines)
    path = 'obj/questions_with_context_test'
    if lines==10:
        path += '_10'
    path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(question_answers_test, f)

    print('saved test data')
    return question_answers_train, question_answers_test

    
