# CIfar-10-image-prediction-with-streamlit-app

<img width="1439" height="803" alt="Screenshot 2025-08-05 at 4 06 36 AM" src="https://github.com/user-attachments/assets/079d66aa-317f-407c-bd97-e4e608ffd7a9" />



# 🚀 CIFAR-10 Image Classifier

This is a web-based image classifier built with **Streamlit** and **TensorFlow**, designed to predict the class of an uploaded image using a pretrained model trained on the **CIFAR-10 dataset**.

---

## 📦 CIFAR-10 Classes

The CIFAR-10 dataset contains 10 different image categories:

- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐶 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

---

## 🌐 Live Demo

> _This project runs locally. To test it, follow the instructions below._

---

## 📁 Project Structure

├── cifar10_model.h5 # (run code and you get this model file)
├── prediction.py # Streamlit app file
├── experiment.ipynb 
├── README.md # Project documentation



## ⚙️ How to Run the Project

### 1. Clone the Repository

```bash
gh repo clone Muahammad-Umair/CIfar-10-image-prediction-with-streamlit-app
cd cifar10-image-classifier
2. Create a Virtual Environment (Optional but Recommended)
bash
Copy
Edit
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
pip install streamlit tensorflow pillow numpy
🚀 Run the App
streamlit run prediction.py
Then open in your browser: http://localhost:8501

🖼️ How to Use
Click "Upload Image"

Upload a PNG, JPG, or JPEG image

The app resizes the image to 32x32

The trained model predicts the class

You’ll see:

✅ Predicted class (e.g., Dog)

📊 Confidence score (e.g., 97.52%)

🖼️ Displayed image (at a smaller size for readability)

📷 Example Output
Uploaded Image: 🐱 Cat

Predicted Class: Cat

Confidence: 97.52%

🧠 Model Information
The model is trained on the CIFAR-10 dataset using a Convolutional Neural Network (CNN)

Accuracy on test set: ~X% (update with actual value if known)
