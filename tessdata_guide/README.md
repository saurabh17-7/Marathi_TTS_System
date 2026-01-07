# Tesseract OCR Setup for Marathi

This project uses Tesseract OCR for extracting Marathi text from images.

## Step 1: Install Tesseract OCR (Windows)
Download and install from:
https://github.com/UB-Mannheim/tesseract/wiki

## Step 2: Download Marathi Language Model
Download `mar.traineddata` from:
https://github.com/tesseract-ocr/tessdata/raw/main/mar.traineddata

## Step 3: Place the File
Copy `mar.traineddata` into:
C:\Program Files\Tesseract-OCR\tessdata\

## Step 4: Verify Installation
Open Command Prompt and run:
tesseract --list-langs

You should see:
mar
