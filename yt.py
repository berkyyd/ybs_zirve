import streamlit as st
import joblib
import pandas as pd

# Modeli ve diğer nesneleri yükle
model_data = joblib.load('best_svm_optimized_balanced.pkl')
best_svm = model_data['model']
vectorizer = model_data['vectorizer']
label_encoder = model_data['label_encoder']

# Streamlit arayüzü
st.title("Yorum Duygu Analizi")
st.write("Bir yorum girin, modelimiz duyguyu tahmin etsin!")

# Kullanıcıdan yorum al
user_comment = st.text_area("Yorumunuzu buraya yazın:", height=150)

# Tahmin butonu
if st.button("Tahmin Yap"):
    if user_comment:
        # Yorumu vektörize et
        X_comment = vectorizer.transform([user_comment])

        # Tahmin yap
        y_pred = best_svm.predict(X_comment)
        predicted_label = label_encoder.inverse_transform(y_pred)[0]

        # Sonucu göster
        st.success(f"Tahmin edilen duygu: **{predicted_label}**")
    else:
        st.error("Lütfen bir yorum girin!")

# Örnek yorumlar için bir sidebar
st.sidebar.header("Örnek Yorumlar")
example_comments = [
    "Çok stresli bir gün geçirdim, her şey üst üste geldi.",
    "Yalnız hissediyorum, kimseyle konuşmak istemiyorum.",
    "Panik atak geçirdim, ne yapacağımı bilemedim."
]
selected_example = st.sidebar.selectbox("Örnek bir yorum seçin:", example_comments)
if st.sidebar.button("Örnek Yorumu Test Et"):
    X_example = vectorizer.transform([selected_example])
    y_pred = best_svm.predict(X_example)
    predicted_label = label_encoder.inverse_transform(y_pred)[0]
    st.sidebar.success(f"Tahmin: **{predicted_label}**")