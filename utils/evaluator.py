import joblib
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
import os
import csv

from utils import load_data


class Evaluator:

    test_data_set=np.delete(np.load('..\\data\\preprocessed\\X_test.npy'),3,axis=1)
    target_values=np.load('..\\data\\preprocessed\\y_test.npy')
    def __init__(self):
        self.cache=[]
        self.col_names=[]
    def scan_models(self):

        models=[i for i in os.listdir('..\\models') if i.endswith('pkl')and "_" in i]
        result_list=[]
        col_names=[]
        for model_file in models:

            model_result= self.evaluate(os.path.join('..\\models',model_file),int(model_file.split(".")[-2].split("_")[-1]))
            model_result['Model Name']=model_file.split("_")[0]
            if not col_names:
                col_names=list(model_result.keys())
            result_list.append(model_result)
        self.cache,self.col_names=result_list,col_names

    def evaluate(self,model_path,stop_index):
        clf = joblib.load(model_path)
        _,self.test_data_set,_,self.target_values=load_data(stop_index)
        accuracy=accuracy_score(clf.predict(self.test_data_set),self.target_values)
        f1=f1_score(clf.predict(self.test_data_set),self.target_values)
        precision=precision_score(clf.predict(self.test_data_set),self.target_values)
        recall=recall_score(clf.predict(self.test_data_set),self.target_values)
        print(clf.__dict__ )
        return {'Accuracy':accuracy,'F1 score':f1,'Precision':precision,'Recall':recall}
    def output_csv(self):
        csv_file = "Evaluation.csv"
        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.col_names)
                writer.writeheader()
                for data in self.cache:
                    writer.writerow(data)
        except IOError:
            print("I\\O error")


if __name__ == '__main__':
    my_evaluator=Evaluator()
    # print(my_evaluator.evaluate('C:\\Users\\ryan_\\PycharmProjects\\ese-417-final-project\\models\\backup\\random-forest_0.pkl',0))
    my_evaluator.scan_models()
    my_evaluator.output_csv()



    

