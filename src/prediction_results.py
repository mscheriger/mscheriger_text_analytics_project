import pickle
from src.ir_method import calculate_accuracy

def get_predictions():
    for f in ("prediction_IR_train.pkl","prediction_IR_test.pkl","prediction_bert_train.pkl","prediction_bert_test.pkl","prediction_bert_train_10.pkl","prediction_bert_test_10.pkl","prediction_bert_train_rouge.pkl","prediction_bert_test_rouge.pkl","prediction_bert_train_10_rouge.pkl","prediction_bert_test_10_rouge.pkl"):
        path = 'obj/' + f
        data = pickle.load(open(path, "rb" ))
        easy = [d for d in data if d['type']=='easy']
        hard = [d for d in data if d['type']=='challenge']

        print("Accuracy for easy questions for {f}: {acc}".format(f=f,acc=calculate_accuracy(easy)))
        print("Accuracy for hard questions for {f}: {acc}".format(f=f,acc=calculate_accuracy(hard)))
        pass
