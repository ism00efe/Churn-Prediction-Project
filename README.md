<<<<<<< HEAD
# End-to-End Churn Prediction Pipeline & API
=======
# Churn Prediction Pipeline
>>>>>>> 821d396 (README dosyası güncellendi)

Telekom musterilerinin churn (abonelik iptali) olasiligini tahmin eden, uctan uca bir makine ogrenmesi projesi.

<<<<<<< HEAD
Projenin temel amacı sadece klasik bir model eğitmek değil; veri temizleme, özellik mühendisliği ve canlıya alma (deployment) süreçlerini modüler, tekrar edilebilir ve otomatize edilebilir bir mimaride kurgulamaktır. Çalışmada endüstri standardı olan Telco Customer Churn veri seti kullanılmıştır.

**🚀 Canlı Demo:** [Hugging Face Space Linkini Buraya Ekle]
=======
Proje; veri temizleme, ozellik muhendisligi, model egitimi, kalibrasyon, API sunumu ve Streamlit arayuzunu tek bir yapida birlestirir.
>>>>>>> 821d396 (README dosyası güncellendi)

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
<<<<<<< HEAD
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
=======
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-orange?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org/)

## Icerik

- [Proje Ozeti](#proje-ozeti)
- [Teknoloji Yigini](#teknoloji-yigini)
- [Klasor Yapisi](#klasor-yapisi)
- [Kurulum](#kurulum)
- [Veri Isleme ve Model Egitimi](#veri-isleme-ve-model-egitimi)
- [FastAPI Servisi](#fastapi-servisi)
- [Streamlit Uygulamasi](#streamlit-uygulamasi)
- [Docker ile Calistirma](#docker-ile-calistirma)
- [Modelleme Notlari](#modelleme-notlari)
- [Katki ve Gelistirme Notlari](#katki-ve-gelistirme-notlari)

## Proje Ozeti

Bu repo, churn tahminini sadece bir notebook deneyi olmaktan cikarip tekrar calistirilabilir bir urunlestirme hattina tasir.

Ana hedefler:

- Temiz ve moduler bir veri-hazirlama yapisi kurmak
- Is odakli bir performans bakisi (net kar senaryolari) eklemek
- Egitilen modeli API ve web arayuz uzerinden servis etmek
- Docker ile ortama bagimli olmayan calistirma saglamak

## Teknoloji Yigini

- Python 3.11+
- pandas, numpy
- scikit-learn
- FastAPI + Uvicorn
- Streamlit
- Docker
- Jupyter Notebook (EDA ve deneysel calismalar)

## Klasor Yapisi

```text
.
|-- App/
|   |-- Main.py              # FastAPI endpoint'leri
|   `-- app.py               # Streamlit arayuzu
|-- Models/
|   `-- Model.pkl            # Egitilmis model (pipeline)
|-- Nootbooks/               # EDA ve modelleme not defterleri
|-- Tests/
|   `-- test_data.py
|-- src/
|   |-- config.py            # Merkezi parametreler
|   |-- Data.py              # Veri temizleme
|   |-- data_loader.py       # Veri okuma ve train/test ayirma
|   |-- evaluate.py          # Esik ve net kar hesaplari
|   |-- features.py          # Encoding, secim, preprocessing
|   `-- Train.py             # Egitim + GridSearchCV + kalibrasyon
|-- requirements.txt
|-- Dockerfile
`-- README.md
```

## Kurulum

### 1) Repoyu klonlayin

```bash
git clone <repo-url>
cd Churn-Prediction-Project-1
```

### 2) Sanal ortam olusturun

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Bagimliliklari yukleyin

```bash
pip install -r requirements.txt
```

## Veri Isleme ve Model Egitimi

Model egitimi icin:

```bash
python -m src.Train
```

Bu adim:

- Ham veriyi yukler
- Temizleme ve ozellik muhendisligi uygular
- Islenmis veriyi `data/processed/cleaned_data.csv` olarak kaydeder
- `LogisticRegression` modeli icin `GridSearchCV` uygular
- Secilen modeli isotonic calibration ile kalibre eder
- Son modeli `Models/Model.pkl` olarak kaydeder

## FastAPI Servisi

API'yi lokal calistirmak icin:

```bash
uvicorn App.Main:app --reload
```

Varsayilan adresler:

- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### Ornek `POST /predict` istegi

```json
{
  "SeniorCitizen": 0,
  "Partner": 1,
  "Dependents": 0,
  "tenure": 24,
  "OnlineBackup": 1,
  "DeviceProtection": 1,
  "OnlineSecurity": 0,
  "TechSupport": 0,
  "Contract": 1,
  "PaperlessBilling": 1,
  "MonthlyCharges": 79.9,
  "InternetService_DSL": 0,
  "InternetService_Fiber optic": 1,
  "InternetService_No": 0,
  "PaymentMethod_Bank transfer (automatic)": 0,
  "PaymentMethod_Credit card (automatic)": 0,
  "PaymentMethod_Electronic check": 1
}
```

Ornek cevap:

```json
{
  "churn_probability": 0.7362,
  "will_churn": true,
  "applied_threshold": 0.4
}
```

## Streamlit Uygulamasi

Web arayuzunu baslatmak icin:

```bash
streamlit run App/app.py
```

Arayuz, kullanicidan musteri bilgilerini alir ve churn olasiligini yuzdesel olarak gosterir.

## Docker ile Calistirma

### 1) Image olusturun

```bash
docker build -t churn-prediction .
```

### 2) Container baslatin

```bash
docker run -p 8000:8000 churn-prediction
```

Sonrasinda API'ye `http://localhost:8000/docs` adresinden erisebilirsiniz.

## Modelleme Notlari

- Temel model: `LogisticRegression`
- Optimizasyon: `GridSearchCV` (F2 odakli skor)
- Kalibrasyon: `CalibratedClassifierCV` (`isotonic`)
- Karar esigi: API katmaninda `0.4`
- Is degeri: `src/evaluate.py` icinde net kar senaryo hesaplari

## Katki ve Gelistirme Notlari

Onerilen gelistirmeler:

- Test kapsamini genisletmek (`Tests/`)
- Veri dogrulama ve schema kontrollerini artirmak
- Model versiyonlama (MLflow / DVC benzeri araclar)
- CI/CD pipeline eklemek

---

Herhangi bir sorunda issue acabilir veya pull request gonderebilirsiniz.
>>>>>>> 821d396 (README dosyası güncellendi)
