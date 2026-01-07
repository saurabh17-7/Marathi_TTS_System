# Marathi OCR → Multi-Dialect Text-to-Speech System

This project converts Marathi text from images, PDFs, and camera input into speech with emotion and dialect support using Python.

---

## 🔹 Features
- Marathi OCR from image, PDF, and camera
- Marathi Text-to-Speech
- Dialects: Standard, Varhadi, Malvani, Ahirani, Kokani
- Emotion-based voice modulation
- User-friendly GUI

---

## 🔹 Technologies Used
- Python 3
- Tesseract OCR
- pytesseract
- gTTS
- OpenCV
- CustomTkinter
- NumPy, SciPy

---

## 🔹 How to Run

### Step 1: Install Python
Install Python 3.9 or above.

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate

✅ Step 3: Install Required Python Libraries
After activating the virtual environment, run:
pip install -r requirements.txt
#This command installs all required Python libraries for the project.

✅ Step 4: Install Tesseract OCR (Marathi Language)
This project uses Tesseract OCR for Marathi text recognition.

Follow the instructions given in:
tessdata_guide/README.md
#Make sure mar.traineddata is placed correctly and OCR is working.

✅ Step 5: Run the Application
After completing all steps above, run the project using:
python marathi_tts.py
The GUI window will open.
You can upload images, PDFs, or use camera input to generate Marathi speech.


⚠️ Known Limitations

Marathi OCR accuracy depends on image quality

Handwritten Marathi text is not supported

Dialect conversion is rule-based


🎓 Academic Note

This project demonstrates an end-to-end pipeline:
OCR → Text Cleaning → Dialect Processing → Emotion-Aware Text-to-Speech



## 👨‍🎓 Authors
- **Saurabh Pawar**
- **Sanyog Swami**
- **Parth Shinde**

Final Year Engineering Project
