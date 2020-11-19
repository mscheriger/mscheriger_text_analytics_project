import pickle

def calculate_accuracy(question_answers):
    #Calculates accuracy - question answers must have a prediction key
    correct = 0
    for qa in question_answers:
        if qa['answerKey'] == qa['prediction']:
            correct += 1
    return correct/len(question_answers)


data = pickle.load(open("prediction_bert_train.pkl", "rb" ))
easy = [d for d in data if d['type']=='easy']
hard = [d for d in data if d['type']=='challenge']

print(calculate_accuracy(easy))
print(calculate_accuracy(hard))

data = pickle.load(open("prediction_bert_test.pkl", "rb" ))
easy = [d for d in data if d['type']=='easy']
hard = [d for d in data if d['type']=='challenge']

print(calculate_accuracy(easy))
print(calculate_accuracy(hard))
