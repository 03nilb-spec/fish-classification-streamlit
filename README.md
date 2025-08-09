# Fish Classifier - Transfer learning + Streamlit
This project classifies fish images into multiple categories using transfer learning with pre-trained deep learning models.
It also includes a Streamlit web app for interactive predictions.

## Features
- Classify fish into multiple categories.
- Uses transfer learning for improved accuracy.
- Upload an image through the Streamlit app to get predictions.
- Displays model confidence scores.
- Saved .h5 models for later use.

## Files 
- app.py                                  
- requirements.txt                        
- README.md                               
- Fish_image_classification.ipynb         
- fish_Mobile_net_V2_1.h5
- history_model_mobile.pkl                

```
Note - Fish_image_classification.ipynb contains 5 pre-trained models, out of which MobileNetV2 is choosed for Streamlit web app.
```

## How to run
### Clone the repository
```
git clone https://github.com/your-username/fish-classifier-cnn-streamlit.git
cd fish-classifier-cnn-streamlit
```

### Install Dependencies
```
pip install -r requirements.txt
```

### Run the Streamlit app
```
streamlit run app.py
```

## App Screenshot
![Streamlit App Screenshot](images/app_screenshot.png)
