import streamlit as st
import joblib
import pandas as pd
import numpy as np
from ast import literal_eval  # Skills liste formatındaysa

# Sayfayı yeniden yüklemeyi engellemek için session state kullanımı
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    if 'selected_state' not in st.session_state:
        st.session_state.selected_state = None

# Veri setini yükle
try:
    data = pd.read_csv("full_df.csv")  # Veri setinin dosya adını güncelle

    # Hata ayıklama
    st.sidebar.write("Veri seti yüklendi. Satır sayısı:", len(data))
    st.sidebar.write("Sütunlar:", data.columns.tolist())

except Exception as e:
    st.error(f"Veri seti yüklenirken hata oluştu: {str(e)}")
    st.stop()

# Modeli ve MLB'yi yükle
try:
    model_data = joblib.load("salary_model_full.pkl")
    pipeline = model_data["model"]
    mlb = model_data["mlb"]
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {str(e)}")
    st.stop()

# Gerekli sütunları seç ve diğerlerini kaldır
required_columns = ["Job", "City", "State", "Work_Type", "Skills"]  # Modelin kullandığı sütunlar
# Veri setindeki sütun isimlerini kontrol et
available_columns = [col for col in required_columns if col in data.columns]
if not all(col in data.columns for col in ["Job", "City", "State", "Work_Type", "Skills"]):
    st.error(f"Eksik sütunlar: {[col for col in required_columns if col not in data.columns]}")
    st.stop()

# Sadece gerekli sütunları al
data = data[available_columns]

# Kategorileri veri setinden çek
cities = sorted(data["City"].dropna().unique().tolist())
states = sorted(data["State"].dropna().unique().tolist())
work_types = sorted(data["Work_Type"].dropna().unique().tolist())
jobs = sorted(data["Job"].dropna().unique().tolist())  # İş unvanları (dropdown için)

# Her eyalet için şehirleri içeren bir sözlük oluştur
state_to_cities = {}
for state_name in states:
    state_cities = data[data["State"] == state_name]["City"].dropna().unique().tolist()
    state_to_cities[state_name] = sorted(state_cities)

# Hata ayıklama - state_to_cities içeriğini kontrol et
st.sidebar.write("### Eyalet-Şehir Eşleşmeleri Kontrolü")
for state_name, city_list in list(state_to_cities.items())[:5]:  # İlk 5 eyaleti göster
    st.sidebar.write(f"{state_name}: {len(city_list)} şehir")
    if len(city_list) > 0:
        st.sidebar.write(f"Örnek şehirler: {city_list[:3]}")

# Skills için benzersiz becerileri çek
skills_list = []
for skills in data["Skills"].dropna():
    if isinstance(skills, str):
        try:
            skills = literal_eval(skills)  # Liste formatı (örneğin, ["Python", "Java"])
        except:
            skills = skills.split(",")  # Virgülle ayrılmış (örneğin, "Python,Java")
    skills_list.extend([skill.strip() for skill in skills])
skills_list = sorted(list(set(skills_list)))

# MLB'nin beceri listesini kullan (daha güvenli)
if hasattr(mlb, "classes_"):
    skills_list = mlb.classes_.tolist()

# Streamlit arayüzü
st.title("Maaş Tahmin Uygulaması")
st.write("İş, şehir, eyalet, çalışma tipi, birim ve becerilerinizi girerek maaş tahmini yapın!")

# Formdan önce eyalet seçimi
selected_state = st.selectbox("Eyalet Seçin", states, key='state_selector')

# Seçilen eyalete göre şehirleri filtrele
filtered_cities = state_to_cities.get(selected_state, [])
st.write(f"Seçilen eyalette {len(filtered_cities)} şehir bulundu.")

# Kullanıcı girdileri
with st.form("input_form"):
    # İş unvanı dropdown olarak
    job = st.selectbox("İş Unvanı", jobs)

    # Form içinde seçilen eyaleti göster (değiştirilemez)
    st.text(f"Seçilen Eyalet: {selected_state}")
    state = selected_state  # Form dışında seçilen eyaleti kullan

    # Filtrelenmiş şehirler
    city = st.selectbox("Şehir", filtered_cities) if filtered_cities else st.selectbox("Şehir", ["Şehir seçin"])

    # Çalışma tipi
    work_type = st.selectbox("Çalışma Tipi", work_types)

    # Beceriler
    selected_skills = st.multiselect("Beceriler (birden fazla seçebilirsiniz)", skills_list)

    # Birim seçeneği ekleyin
    unit = st.selectbox("Birim", ["Year", "Month", "Hour"])

    # Formu gönder
    submitted = st.form_submit_button("Tahmin Yap")

# Tahmin işlemi
if submitted:
    try:
        # Eğer şehir seçilmediyse veya "Şehir seçin" ise hata ver
        if not city or city == "Şehir seçin":
            st.error("Lütfen geçerli bir şehir seçin.")
            st.stop()

        # Tahmin detaylarını göster
        st.write("### Seçilen Parametreler")
        st.write(f"- Eyalet: {state}")
        st.write(f"- Şehir: {city}")
        st.write(f"- İş: {job}")
        st.write(f"- Çalışma Tipi: {work_type}")
        st.write(f"- Birim: {unit}")
        st.write(f"- Seçilen Beceriler: {', '.join(selected_skills) if selected_skills else 'Yok'}")

        # Kullanıcı girdilerini bir veri çerçevesine dönüştür
        input_data = pd.DataFrame({
            "Job": [job],
            "City": [city],
            "State": [state],
            "Work_Type": [work_type],
            "Unit": [unit]  # Burada `Unit` sütunu ekleniyor
        })

        # Becerileri MLB ile encode et
        skills_encoded = mlb.transform([selected_skills])
        if isinstance(skills_encoded, np.ndarray):
            skills_encoded = skills_encoded[0]  # İlk satırı al
        else:
            skills_encoded = skills_encoded.toarray()[0]  # Sparse matris ise çevir

        skills_columns = mlb.classes_
        skills_df = pd.DataFrame([skills_encoded], columns=skills_columns)

        # Girdileri birleştir
        final_input = pd.concat([input_data, skills_df], axis=1)

        # Tahmin yap
        prediction = pipeline.predict(final_input)

        # Sonucu göster
        st.success("Tahmin Başarılı!")
        st.write(f"Tahmini Maaş: {prediction[0]:,.2f} $/Year")  # Çıktı formatını güncelle
    except Exception as e:
        st.error(f"Tahmin sırasında hata oluştu: {str(e)}")
        st.error(f"Hata detayı: {str(e)}")

# Ek bilgi
st.markdown(
    "**Not:** Bu uygulama, yüklenen modele göre tahmin yapar. Doğru sonuçlar için lütfen geçerli bilgiler girin.")