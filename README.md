# Real-Time Speech Enhancement System for Classrooms

![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)

---

## 📚 Overview

This project presents a **Real-Time Speech Enhancement System** designed to improve voice clarity in noisy classroom environments.  
Leveraging **deep learning** techniques and a **PercepNet-inspired architecture**, the system transforms noisy speech into clean, natural-sounding audio by processing spectrogram representations.

Built using **PyTorch** and **Torchaudio**, the system is optimized for real-time performance, making it suitable for applications like virtual classrooms, hearing aids, and IoT-based communication devices.

---

## 🚀 Features
- **Speech Enhancement:** Suppresses background noise while preserving natural speech quality.
- **Real-Time Capable:** Fast and efficient model suitable for live audio processing.
- **PercepNet-based Architecture:** Combines convolutional and recurrent layers for effective time-frequency feature learning.
- **STFT and ISTFT:** Uses Short-Time Fourier Transform for feature extraction and reconstruction.

---

## 🛠️ Technologies Used
- **Python 3.8+**
- **PyTorch**
- **Torchaudio**
- **NumPy, SciPy**
- **Matplotlib**

---

## 📈 Results
- **Enhanced clarity and intelligibility** in noisy environments.
- **Waveform visualization** shows significant noise reduction after enhancement.
- Suitable for **real-time deployments** with minimal latency.

---

## 📦 Dataset

The system uses paired **clean** and **noisy speech samples** from publicly available datasets, ensuring a diverse set of noise conditions for training.

> 🔗 **Dataset Example:** [VoiceBank + DEMAND](https://datashare.ed.ac.uk/handle/10283/2791)  
(*You can upload your dataset link if needed.*)

---

## 📂 Project Structure
```plaintext
├── data/
│   ├── clean/
│   ├── noisy/
├── CODE.py
├── README.md
```

---


## 🏁 How to Run
```bash
# Clone the repository
git clone https://github.com/your_username/your_repo_name.git
cd your_repo_name

# Install dependencies
pip install -r requirements.txt

# Train or test the model
python main.py
```

---

## 🔥 Future Work
- Integrate **advanced phase reconstruction** techniques for better audio quality.
- Extend the model to support **multi-microphone input** (beamforming).
- Deploy the system on **embedded devices** (e.g., Raspberry Pi, IoT nodes).
- Implement **objective evaluation metrics** like PESQ, STOI, and SNR improvements.

---

## 📣 Acknowledgments
- [VoiceBank-DEMAND Dataset](https://datashare.ed.ac.uk/handle/10283/2791)
- [PercepNet: Perceptually Motivated Real-Time Speech Enhancement](https://arxiv.org/abs/2008.04560)
- [PyTorch](https://pytorch.org/)
- [Torchaudio](https://pytorch.org/audio/stable/index.html)

---

# 🎧 Thank you for visiting! Please ⭐ the repo if you find it helpful!

