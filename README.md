# Churn Prediction Pipeline & API

Bu proje, telekom müşterilerinin servisi terk etme (churn) olasılığını tahmin eden ve bu tahminleri bir REST API / Web arayüzü ile sunan uçtan uca bir makine öğrenmesi hattıdır. 

Projenin temel amacı sadece bir model eğitmek değil; veri temizleme, özellik mühendisliği ve canlıya alma (deployment) süreçlerini modüler, tekrar edilebilir ve otomatize edilebilir bir mimaride kurgulamaktır.

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

## Mimari ve Tasarım Kararları

Geliştirme sürecinde "Jupyter Notebook karmaşasından" kaçınmak ve kodu ürünleşmeye hazır (production-ready) hale getirmek için şu mimari kararları uyguladım:

* **Modüler Yapı (`src/`):** Veri işleme, özellik mühendisliği ve eğitim adımlarını tek bir dosyaya yığmak yerine fonksiyonel olarak böldüm.
* **Merkezi Konfigürasyon (`config.py`):** Korelasyon sınırları (threshold=0.10) ve silinecek sütunlar gibi hiperparametreleri tek bir noktadan yöneterek hard-code kullanımını engelledim.
* **İş Mantığı Odaklı Değerlendirme (`evaluate.py`):** Modelin başarısını sadece standart metriklerle (Accuracy, F1) değil; V_cost ve C_cost gibi parametreler kullanarak iyimser/gerçekçi net kâr (net profit) senaryolarıyla ölçtüm.
* **API ve Konteynerizasyon:** Eğitilen model (Logistic Regression + Isotonic Calibration), FastAPI kullanılarak bir servise dönüştürüldü ve Docker ile platform bağımsız çalışabilir hale getirildi.

## Dizin Yapısı

```text
├── App/                    # Streamlit ve FastAPI arayüz kodları
├── Models/                 # Eğitilmiş ve kalibre edilmiş .pkl modeli
├── data/                   # Ham ve işlenmiş veri setleri (raw/ & processed/)
├── src/                    
│   ├── config.py           # Proje parametreleri ve eşik değerler
│   ├── Data.py             # Veri temizleme adımları
│   ├── features.py         # Encoding ve ColumnTransformer işlemleri
│   ├── evaluate.py         # Confusion matrix tabanlı kâr hesaplamaları
│   ├── data_loader.py      # Veri okuma ve train/test ayırma
│   └── Train.py            # GridSearchCV ile model eğitimi
├── app.py                  # Streamlit UI
├── Main.py                 # FastAPI Endpoint'leri
└── Dockerfile              # Konteyner imaj tanımları
