import os
import glob
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

def main():
    print("--- 1. Sistem Kontrolü ---")
    ultralytics.checks()
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Kullanılan donanım: {device}\n")

    DATASET_DIR = r"C:\Users\omerf\Yolo Projects\dataset"
    RUNS_DIR = r"C:\Users\omerf\Yolo Projects\runs"

    if not os.path.exists(DATASET_DIR):
        print(f"[HATA] Belirtilen veri seti yolu bulunamadı: {DATASET_DIR}")
        print("Lütfen klasör adını ve konumunu kontrol edin.")
        return

    print("--- 2. Model Eğitimi Başlıyor ---")
    model = YOLO("yolo11n-cls.pt")

    model.train(
        data=DATASET_DIR,
        imgsz=480,
        epochs=2,
        batch=16,
        workers=2,
        device=device,
        project=RUNS_DIR,       
        name="Tumor_Run"
    )

    print("\n--- 3. Test ve Çıkarım Aşaması ---")
    weights_path = os.path.join(RUNS_DIR, "Tumor_Run", "weights", "best.pt")

    if not os.path.exists(weights_path):
        print(f"[UYARI] Ağırlık dosyası bulunamadı: {weights_path}")
        return

    trained_model = YOLO(weights_path)

    test_img_path = r"C:\Users\omerf\Yolo Projects\dataset\test\no_tumor\image-187-_jpg.rf.7afa6c4b99ea6e1977321f5ae7debcaf.jpg"

    if not os.path.exists(test_img_path):
        print(f"[UYARI] Belirtilen görsel bulunamadı: {test_img_path}")
        return

    print(f"Test edilen görsel: {test_img_path}")

    results = trained_model(test_img_path)

    class_dict = results[0].names
    probs = results[0].probs.data.tolist()

    predicted_class_idx = np.argmax(probs)
    predicted_class = class_dict[predicted_class_idx]
    confidence = probs[predicted_class_idx]

    print(f"Sonuç: {predicted_class} (Güven Oranı: %{confidence * 100:.2f})")

    img = cv2.imread(test_img_path)
    text = f"{predicted_class}: {confidence:.2%}"
    cv2.putText(img, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Tumor Teshis Sonucu", img)
    print("Pencereyi kapatmak için görsel üzerindeyken herhangi bir tuşa basın...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    save_dir = r"C:\Users\omerf\Yolo Projects\tümör_project\images"
    os.makedirs(save_dir, exist_ok=True)  

    file_name = os.path.basename(test_img_path)
    save_path = os.path.join(save_dir, f"tahmin_{file_name}")

    cv2.imwrite(save_path, img)
    print(f"Görsel başarıyla kaydedildi: {save_path}")

if __name__ == '__main__':
    main()