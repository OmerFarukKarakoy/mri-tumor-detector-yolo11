# 🧠 MRI Brain Tumor Detection & Classification with YOLO11

![YOLO11](https://img.shields.io/badge/Model-YOLO11--cls-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Language-Python%203.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Task](https://img.shields.io/badge/Task-Medical%20Imaging%20Classification-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**MRI Tumor Detector YOLO11**, beyin Manyetik Rezonans Görüntüleme (MR / MRI) taramaları üzerinden tümör varlığını otomatik olarak tespit eden ve 4 farklı klinik sınıfa ayıran derin öğrenme tabanlı bir bilgisayarlı görü projesidir. 

Proje, Ultralytics'in en son nesil mimarisi olan **YOLO11 (Classification)** kullanılarak eğitilmiştir. Model, MR görüntülerini piksel düzeyinde analiz eder, sınıf olasılıklarını (`confidence`) hesaplar ve tanı çıktısını doğrudan görüntü üzerine etiketleyerek kaydeder.

✅ **Mimari:** YOLO11-cls (YOLOv11 Nano Sınıflandırma Omurgası)  
✅ **Hedef:** Erken teşhise yardımcı radyolojik karar destek sistemi  
✅ **Uygulama Alanı:** Medikal Görüntü İşleme, Derin Öğrenme, Klinik Yapay Zeka

---

## 🔬 Sınıflandırma Kategorileri & Klinik Kapsam

Model, beyin MR taramalarında aşağıdaki 4 farklı sınıfı yüksek doğruluk ve güven skoruyla ayırt edebilmektedir:

1. **Glioma Tumor (Gliom Tümörü):** Beyin ve omurilikteki destek doku olan glial hücrelerden kaynaklanan primer beyin tümörleri.
2. **Meningioma Tumor (Menenjiyom Tümörü):** Beyni ve omuriliği çevreleyen zar tabakasında (meninks) gelişen tümörler.
3. **Pituitary Tumor (Hipofiz Bezi Tümörü):** Beynin tabanında yer alan hipofiz bezinde gelişen ve hormonal dengeleri etkileyen kitleler.
4. **No Tumor (Tümörsüz / Sağlıklı Doku):** Herhangi bir lezyon veya tümör bulgusuna rastlanmayan normal beyin dokusu.

---

## 🖥️ Örnek Teşhis Çıktıları (Visual Results)

Aşağıdaki çıktılar, test veri kümesinde daha önce modele gösterilmemiş MR görüntüleri üzerinde elde edilen gerçek zamanlı model çıkarım (inference) sonuçlarıdır:

### 1. Glioma Tumor & Meningioma Tumor

| Glioma Tumor Tespiti | Meningioma Tumor Tespiti |
| :---: | :---: |
<img width="350" alt="meningioma-tumor" height="640" src="https://github.com/user-attachments/assets/3b6fe6af-a0c7-44ec-bb27-884a5686d332" />
<img width="350" alt="g<img width="640" height="640" src="https://github.com/user-attachments/assets/bd0868f5-9d40-465c-9355-fe4a96a94248" />
lioma-tumor" height="640" alt="tahmin_m-196-_jpg rf fdd350ea34acf41536b7ed4be4f8c28c" src="https://github.com/user-attachments/assets/524e5482-610b-42ae-b722-405b11bec1c9" />
| *Görüntü analizinde Glioma lezyonu tespit edilmiş ve güven skoruyla işlenmiştir.* | *Meningioma zar tümörü tespiti ve güven skoru.* |


---

### 2. Pituitary Tumor & No Tumor (Sağlıklı)

| Pituitary Tumor Tespiti | No Tumor (Sağlıklı Doku) |
| :---: | :---: |
| <img width="350" alt="pituitary-tumor" src="https://github.com/user-attachments/assets/cabd7abf-af48-4ded-b8e0-31f3cdc23065" /> 
| <img width="350" alt="no-tumor" src="https://github.com/user-attachments/assets/78a3c09b-d1e3-4161-ad72-4c22e699e1f3" />> |
| *Hipofiz bezi lezyonunun tespiti ve sınıflandırılması.* | *Tümör bulgusu içermeyen sağlıklı MR taramasının doğrulanması.* |


---

## ⚙️ Temel Özellikler

- **Modern Mimari:** Ultralytics YOLO11 sınıflandırma omurgası ile optimize edilmiş ağırlıklar.
- **Güvenli Çıkarım Hattı (Inference Pipeline):** Test görüntüsü üzerinde anlık tahmin üretir, sınıf olasılıklarını çıkarır ve en yüksek güven skorunu hesaplar.
- **Görselleştirme & Dışa Aktarma (Export):** OpenCV entegrasyonu ile sınıf adı ve olasılık oranını doğrudan resme işleyip `tahmin_sonuclari` klasörüne otomatik kaydeder.
- **Hafif ve Hızlı:** Düşük gecikme süresiyle CPU ve GPU ortamlarında hızlı çıkarım performansı.

---

## 🧰 Teknoloji Yığını

| Bileşen | Teknoloji | Açıklama |
| :--- | :--- | :--- |
| **Model / Mimari** | Ultralytics YOLO11 | Görüntü sınıflandırma modeli (`yolo11n-cls`) |
| **Derin Öğrenme** | PyTorch | Model eğitimi ve tensör hesaplamaları |
| **Görüntü İşleme** | OpenCV (`cv2`) | Görsel okuma, metin ekleme ve diske kaydetme |
| **Veri Manipülasyonu** | NumPy | Olasılık dizisi analizi ve argmax indeksleme |
| **Programlama Dili** | Python 3.10+ | Uçtan uca proje omurgası |

---

## 🚀 Kurulum ve Çalıştırma

1. Repoyu Klonlayın


```bash
git clone [https://github.com/OmerFarukKarakoy/mri-tumor-detector-yolo11.git](https://github.com/OmerFarukKarakoy/mri-tumor-detector-yolo11.git)
cd mri-tumor-detector-yolo11<img width="640" height="640" alt="tahmin_gg-52-_jpg rf 078cffb6ed2cbf35c6c0955fbba271e7" src="https://github.com/user-attachments/assets/bb10d2f1-4385-453e-b452-15776e038f4a" />


2. Gereksinimleri Yükleyin

```bash
   pip install ultralytics opencv-python numpy torch
   ```

3. Modeli Test Edin (Inference)
  main.py dosyasını açıp test etmek istediğiniz MR görüntüsünün yolunu belirleyin ve çalıştırın:
```bash
   python main.py
   ```

## 👨‍💻 Developer

**Ömer Faruk Karaköy**    
🌐 GitHub: [github.com/OmerFarukKarakoy](https://github.com/OmerFarukKarakoy)  
📧 Mail: omerfarukkarakoy@hotmail.com


## Join the community

Join our community of developers creating universal apps.

- [Expo on GitHub](https://github.com/expo/expo): View our open source platform and contribute.
- [Discord community](https://chat.expo.dev): Chat with Expo users and ask questions.
