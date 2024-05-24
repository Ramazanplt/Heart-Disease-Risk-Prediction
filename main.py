#Kütüphaneleri import edelim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Veri setini yükleyelim
doc = 'C:/Users/TR/Desktop/heart_disease_uci.csv'
data = pd.read_csv(doc)

#İlk birkaç satırı görüntüleyelim
print(data.head())

#Veri seti hakkında genel bilgileri görüntüleyelim
print(data.info())

#Eksik veri olup olmadığını kontrol edelim
print(data.isnull().sum())

#Veri setinin istatistiksel özetini görüntüleyelim
print(data.describe())
print(data.columns)

#Hedef değişkenin dağılımını gösterelim
if 'num' not in data.columns:
    print("Hedef değişken 'num' veri setinde bulunamadı. Mevcut sütun adlarını kontrol edin.")
else:
    sns.countplot(x='num', data=data)
    plt.show()

    #Yaş değişkeninin dağılımını gösterelim
    sns.histplot(data['age'], kde=True)
    plt.show()

    #Hedef değişkenine göre yaş değişkeninin dağılımını gösterelim
    sns.boxplot(x='num', y='age', data=data)
    plt.show()

    #Tüm kategorik sütunları (one-hot encoding) ile dönüştürme
    data = pd.get_dummies(data, columns=['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'])

    #Bağımsız ve bağımlı değişkenleri ayırma
    X = data.drop('num', axis=1)
    y = data['num']

    #Eksik değerleri doldurma veya atma işlemleri
    X.fillna(X.mean(), inplace=True)

    #Veriyi eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Modeli eğitme ve değerlendirme
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

