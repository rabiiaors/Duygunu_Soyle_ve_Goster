# Duygu Analizi Uygulaması

Bu Flask uygulaması, yüklenen ses ve video dosyalarında duygu tanıma işlemi gerçekleştirir. Ses duygu tanıma için Wav2Vec2 modelini, video analizi için ise önceden eğitilmiş bir yüz duygu tanıma modelini kullanır.

## Özellikler

- Wav2Vec2 modelini kullanarak ses duygu tanıma.
- Yüz duygu tanıma modelini kullanarak video duygu tanıma.
- Real time bir şekilde duygu analizinin yapılması

## Kurulum

### Gereksinimler
- Python 3.7+
- Sanal ortam aracı (isteğe bağlı ancak önerilir)


### Adımlar
1. **Depoyu klonlayın:**
   ```bash
   git clone https://github.com/rabiiaors/Duygunu_Soyle_ve_Goster
   cd Duygunu_Soyle_ve_Goster
   
    ya da direkt indirin


### Sanal ortam oluşturmak için Adımlar
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


### Gerekli kütüphalerin yüklenmesi
pip install -r requirements.txt


### Programın çalıştırılması
python app2.py


### Aşağıdaki link ile proje sayfasına ulaşılması
http://127.0.0.1:5000/


 
 
 ### Proje yapısı
 
 emotion-analysis-app/
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   └── index.html
│
├── uploads/
│
├── .gitignore
├── app.py
├── facialemotionmodel.h5
├── facialemotionmodel.json
├── README.md
├── requirements.txt
└── venv/

