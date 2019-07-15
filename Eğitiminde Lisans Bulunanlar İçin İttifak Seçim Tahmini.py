import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("yonelimfinal.csv")


#Seçilen zaman bir etken olamayacağı için veri dizisinden sütun ayırılıyor.
#Üniversitedeki öğrenciler dikkate alınacağı için yaş aralığı sınırlanıyor.
#Eğitim durumu lisans ve önlisans olanlar dikkate alınacağı için eğitim lisans ve önlisans olanlar dikkate alınıyor.
#Kadın erkek farkı gözetmeksizin yapılıyor.

df_genc_zamansiz = df.drop('Timestamp',axis=1)
df_genc_zamansiz = df_genc_zamansiz[(df_genc_zamansiz['Yas']=='18-30')]
df_genc_zamansiz = df_genc_zamansiz.drop('Yas',axis=1)
df_genc_zamansiz = df_genc_zamansiz.drop('Cinsiyet',axis=1)
df_genc_zamansiz1 = df_genc_zamansiz[(df_genc_zamansiz['Egitim']=='Lisans')]
df_genc_zamansiz2 = df_genc_zamansiz[(df_genc_zamansiz['Egitim']=='Ön Lisans')]
df_genc_zamansiz = df_genc_zamansiz1.append(df_genc_zamansiz2)
df_genc_zamansiz = df_genc_zamansiz.drop('Egitim',axis=1)
df_genc_zamansiz = df_genc_zamansiz.drop('Bolge',axis=1)
del df_genc_zamansiz2,df_genc_zamansiz1

#Hayır'a False Evet'e True değeri atanıyor.
#Partilere isim yerine sayısal bir değer atanıyor.

def cevap_durumu(cevap):
    return (cevap == 'Evet')

df_son = df_genc_zamansiz
df_son['soru1'] = df['soru1'].apply(cevap_durumu)
df_son['soru2'] = df['soru2'].apply(cevap_durumu)
df_son['soru3'] = df['soru3'].apply(cevap_durumu)
df_son['soru4'] = df['soru4'].apply(cevap_durumu)
df_son['soru5'] = df['soru5'].apply(cevap_durumu)
df_son['soru6'] = df['soru6'].apply(cevap_durumu)
df_son['soru7'] = df['soru7'].apply(cevap_durumu)
df_son['soru8'] = df['soru8'].apply(cevap_durumu)
df_son['soru9'] = df['soru9'].apply(cevap_durumu)
df_son['soru10'] = df['soru10'].apply(cevap_durumu)

#Diğer partiler dikkate alınmıyor.
df_son=df_son[df_son.parti != 'DIĞER']
# Encoding categorical data
x = {'AKP': 1, 'MHP': 1,'IYI PARTI' : 0, 'CHP' : 0, 'HDP' : 0}

df_son = df_son.replace(x)

#partiler ve sorular birbirinden ayrılıyor.
df_son_parti=df_son.iloc[0:,10]
df_son_soru=df_son.iloc[:,:-1]

class Functions_Of_Predict:

    def __init__(self):

        questions_list=["Ekonomik durumumuzun iyi olduğunu düşünüyor musunuz?",
                "Eğitimde reform istiyor musunuz?",
                "Devlet dairelerinin özelleştirilmesini istiyor musunuz?",
                "Devletin bazı suçlar için ölüm cezası verebilmesini istiyor musunuz?",
                "Habercilerimizi yeteri kadar tarafsız buluyor musunuz?",
                "22:00'dan sonra içki yasağını destekliyor musunuz?",
                "Laik ülkede yaşamak istiyor musunuz?","Kürtaj yasağını destekliyor musunuz?",
                "Ohalin özgürlükleri sınırladığını düşünüyor musunuz?",
                "Parlementer sisteme geri dönülmesini istiyor musunuz?"]

        self.questions_list = questions_list

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(df_son_soru, df_son_parti, test_size = 0.15, random_state =1)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get_answers(self):

        answer_list = []

        for recorder in range(0,10):
            answer = input(self.questions_list[recorder])

            if answer == "1":

                answer = True
            else:

                answer = False

            answer_list.append(answer)

        self.answer_list = answer_list

        return self.answer_list

    def Choice_Predict(self,df_forcoloumns):
        answertotable=df_forcoloumns[0:0]
        listOfSeries = [pd.Series(self.answer_list, index=answertotable.columns )]
        answertotable=answertotable.append(listOfSeries)
        self.answertotable=answertotable
        return answertotable


    def Model_Comparison(self):
        #Sınıflandırma Modellerine Ait Kütüphaneler
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

        models = []
        models.append(('Naive Bayes', GaussianNB()))
        models.append(('Logistic Regression', LogisticRegression()))
        models.append(('K-NN', KNeighborsClassifier()))
        models.append(('SVM', SVC()))
        models.append(('RandomForestClassifier', RandomForestClassifier()))
        models.append(('AdaBoostClassifier', AdaBoostClassifier()))
        models.append(('BaggingClassifier', BaggingClassifier()))

        from sklearn.metrics import classification_report
        from sklearn import metrics
        from sklearn.metrics import roc_curve, auc
        from sklearn.feature_selection import RFE
        # Modelleri test edelim
        for name, model in models:
            model = model.fit(self.X_train, self.y_train)
            Y_pred = model.predict(self.X_test)
            Y_real_pred=model.predict(self.answertotable)
            print(Y_pred)

            #Accuracy değeri gör

            print("%s -> ACC: %%%.2f" % (name,metrics.accuracy_score(self.y_test, Y_pred)*100))
            print(Y_real_pred)

            #Confusion matris görmek için aşağıdaki kod satırlarını kullanabilirsiniz
            #report = classification_report(y_test, Y_pred)
            #print(report)

    def BestofAlgorithms_Iteration_11times(self):

        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
        from sklearn.metrics import classification_report
        from sklearn import metrics
        from sklearn.metrics import roc_curve, auc
        from sklearn.feature_selection import RFE
        models = []

        models.append(('RandomForestClassifier', RandomForestClassifier()))
        models.append(('AdaBoostClassifier', AdaBoostClassifier()))
        models.append(('BaggingClassifier', BaggingClassifier()))

        for name, model in models:
            predict_list = []

            model = model.fit(self.X_train, self.y_train)
            Y_pred = model.predict(self.X_test)
            Y_real_pred=model.predict(self.answertotable)

            for x in range(0,11):

                model = model.fit(self.X_train,self.y_train)
                Y_pred = model.predict(self.X_test)
                Y_real_pred = model.predict(self.answertotable)
                predict_list.append(Y_real_pred)

            if (predict_list.count(1)<predict_list.count(0)):
                print(name,"Millet İttifakı")

            else:
                print(name,"Cumhur İttifakı")



Do_it=Functions_Of_Predict()
print(Do_it.get_answers())
print("\n")
print(Do_it.Choice_Predict(df_son_soru))
print("\n")
Do_it.BestofAlgorithms_Iteration_11times()
print("\n")
Do_it.Model_Comparison()

print("Not: En Doğru Sonuçlar Adaboostla bulundu.")
