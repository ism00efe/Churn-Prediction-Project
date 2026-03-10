# 1. Hafif bir Python imajı seçiyoruz
FROM python:3.11-slim

# 2. Çalışma dizini oluşturuyoruz
WORKDIR /app

# 3. Önce sadece requirements.txt kopyalayıp kuruyoruz. 
# Bu sayede kodun değişse bile kütüphaneler tekrar tekrar yüklenmez (Cache avantajı)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Proje klasörlerini Docker içine kopyalıyoruz
COPY App/ ./App/
COPY Models/ ./Models/
COPY src/ ./src/

# 5. Render'ın beklediği portu açıyoruz
EXPOSE 8000

# 6. Uygulamayı başlatıyoruz
# Dosya yapına göre 'App.Main:app' kısmını kontrol etmelisin
CMD ["uvicorn", "App.Main:app", "--host", "0.0.0.0", "--port", "8000"]