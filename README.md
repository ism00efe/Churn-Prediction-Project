# End-to-End Churn Prediction Pipeline & API

Bu proje, telekom müşterilerinin servisi terk etme (churn) olasılığını tahmin eden ve bu tahminleri bir REST API / Web arayüzü ile sunan uçtan uca bir makine öğrenmesi hattıdır. 

Projenin temel amacı sadece klasik bir model eğitmek değil; veri temizleme, özellik mühendisliği ve canlıya alma (deployment) süreçlerini modüler, tekrar edilebilir ve otomatize edilebilir bir mimaride kurgulamaktır. Çalışmada endüstri standardı olan Telco Customer Churn veri seti kullanılmıştır.

**🚀 Canlı Demo:** [Hugging Face Space Linkini Buraya Ekle]

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/)

## 📸 Arayüz (Streamlit UI)

![Churn App](arayuz_gorseli_linki.png) *(Buraya repoya yüklediğin arayüz görselinin yolunu koy)*

## 🏗️ Mimari ve Tasarım Kararları

Geliştirme sürecinde araştırma kodlarını (Jupyter Notebook) ürünleşmeye hazır (production-ready) hale getirmek için şu mimari kararlar uygulanmıştır:

* **Modüler Pipeline (`src/`):** Veri işleme, özellik mühendisliği ve eğitim adımları tek bir dosyaya yığılmak yerine fonksiyonel olarak izole edilmiştir.
* **Merkezi Konfigürasyon (`config.py`):** Korelasyon sınırları (threshold=0.10) ve silinecek sütunlar gibi hiperparametreler tek bir noktadan yönetilerek hard-code kullanımı engellenmiştir.
* **İş Mantığı Odaklı Değerlendirme:** Modelin başarısı sadece standart metriklerle değil; müşteri kazanım ve kayıp maliyetleri (V_cost, C_cost) üzerinden oluşturulan kâr/zarar senaryolarıyla ölçülmüştür. Bu nedenle model, *Recall* değerini maksimize edecek şekilde (F2-Score) optimize edilmiştir.
* **Kalibrasyon:** Model çıktıları sadece 0 ve 1'den ibaret olmaması ve gerçek olasılık değerleri üretmesi için `CalibratedClassifierCV` (Isotonic) kullanılarak kalibre edilmiştir.

## 📊 Model Performansı

Logistic Regression modeli GridSearchCV ile hiperparametre optimizasyonuna sokulmuş ve aşağıdaki sonuçlar elde edilmiştir:

* **Accuracy:** [% XX]
* **Recall (Churn Sınıfı İçin):** [% XX] -> *(Özellikle bu metrik yüksek tutulmaya çalışılmıştır)*
* **F2-Score:** [% XX]

*Not: Detaylı Keşifsel Veri Analizi (EDA) ve model denemeleri `01_eda.ipynb` ve `02_modeling.ipynb` dosyalarında bulunabilir.*

## 📂 Dizin Yapısı

```text
├── App/                    # Streamlit ve FastAPI arayüz kodları
├── Models/                 # Eğitilmiş ve kalibre edilmiş model (.pkl)
├── data/                   # Ham ve işlenmiş veri setleri
├── src/                    
│   ├── config.py           # Proje parametreleri ve maliyet senaryoları
│   ├── Data.py             # Veri temizleme orkestrasyonu
│   ├── features.py         # Encoding ve bellek optimizasyonu
│   ├── evaluate.py         # Confusion matrix tabanlı finansal hesaplamalar
│   └── Train.py            # Modelin eğitilmesi ve dışa aktarılması
├── app.py                  # Streamlit UI
├── Main.py                 # FastAPI Endpoint'leri
└── Dockerfile              # Konteyner imaj tanımları
