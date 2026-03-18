import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Veriyi yükle ve modeli eğit
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df = df.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"])
df["Age"] = df["Age"].fillna(df["Age"].median())
df = df.dropna()
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]
model = LogisticRegression()
model.fit(X, y)

# Arayüz
st.title("🚢 Titanic Hayatta Kalma Tahmini")
st.write("Bilgilerini gir, Titanik'te hayatta kalır mıydın?")

pclass = st.selectbox("Bilet Sınıfı", [1, 2, 3])
sex = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
age = st.slider("Yaş", 1, 80, 25)
fare = st.slider("Bilet Ücreti ($)", 1, 500, 50)

sex_val = 0 if sex == "Kadın" else 1

if st.button("Tahmin Et"):
    sonuc = model.predict([[pclass, sex_val, age, fare]])
    if sonuc[0] == 1:
        st.success("🎉 Hayatta Kalırdın!")
    else:
        st.error("💀 Hayatta Kalamazdın!")