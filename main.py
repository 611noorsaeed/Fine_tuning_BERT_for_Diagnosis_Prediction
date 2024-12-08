# pip install fastapi uvicorn torch transformers PyPDF2 aiofiles scikit-learn nltk python-multipart

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import PyPDF2
from io import BytesIO
import pickle
from fastapi import Request
import numpy as np
import re
from nltk.corpus import stopwords

# Make sure to download stopwords from nltk
import nltk
nltk.download('stopwords')


# Load the model, tokenizer, and label encoder
model = BertForSequenceClassification.from_pretrained('./patient_model')
tokenizer = BertTokenizer.from_pretrained('./patient_model')
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))

# FastAPI instance
app = FastAPI()

# Set up templates and static file directory
templates = Jinja2Templates(directory="templates")


# Disease data
disease_data = {
    "Peptic Ulcer Disease": {
        "description": "A sore that develops on the lining of the esophagus, stomach, or small intestine.",
        "medicines": ["Omeprazole", "Pantoprazole", "Ranitidine", "Esomeprazole", "Amoxicillin"],
        "specialists": ["Gastroenterologist", "General Physician", "Internal Medicine Specialist"]
    },
    "Type 2 Diabetes Mellitus": {
        "description": "A chronic condition that affects the way the body processes blood sugar (glucose).",
        "medicines": ["Metformin", "Glipizide", "Insulin", "Sitagliptin", "Canagliflozin"],
        "specialists": ["Endocrinologist", "Diabetologist", "Nutritionist"]
    },
    "Acute Myocardial Infarction": {
        "description": "A medical emergency where the blood flow to the heart is blocked.",
        "medicines": ["Aspirin", "Clopidogrel", "Statins", "Beta Blockers", "ACE Inhibitors"],
        "specialists": ["Cardiologist", "Emergency Medicine Specialist"]
    },
    "Chronic Obstructive Pulmonary Disease": {
        "description": "A group of lung diseases that block airflow and make breathing difficult.",
        "medicines": ["Tiotropium", "Albuterol", "Ipratropium", "Fluticasone", "Salmeterol"],
        "specialists": ["Pulmonologist", "General Physician", "Respiratory Therapist"]
    },
    "Cerebrovascular Accident (Stroke)": {
        "description": "A condition caused by the interruption of blood flow to the brain.",
        "medicines": ["Alteplase", "Aspirin", "Clopidogrel", "Warfarin", "Atorvastatin"],
        "specialists": ["Neurologist", "Rehabilitation Specialist", "Neurosurgeon"]
    },
    "Deep Vein Thrombosis": {
        "description": "A blood clot forms in a deep vein, usually in the legs.",
        "medicines": ["Warfarin", "Heparin", "Apixaban", "Dabigatran", "Rivaroxaban"],
        "specialists": ["Hematologist", "Vascular Surgeon", "Cardiologist"]
    },
    "Chronic Kidney Disease": {
        "description": "The gradual loss of kidney function over time.",
        "medicines": ["Erythropoietin", "Phosphate Binders", "ACE Inhibitors", "Diuretics", "Calcitriol"],
        "specialists": ["Nephrologist", "Dietitian", "Internal Medicine Specialist"]
    },
    "Community-Acquired Pneumonia": {
        "description": "A lung infection acquired outside of a hospital setting.",
        "medicines": ["Amoxicillin", "Azithromycin", "Clarithromycin", "Ceftriaxone", "Levofloxacin"],
        "specialists": ["Pulmonologist", "Infectious Disease Specialist", "General Physician"]
    },
    "Septic Shock": {
        "description": "A severe infection leading to dangerously low blood pressure.",
        "medicines": ["Norepinephrine", "Vancomycin", "Meropenem", "Hydrocortisone", "Dopamine"],
        "specialists": ["Intensivist", "Infectious Disease Specialist", "Emergency Medicine Specialist"]
    },
    "Rheumatoid Arthritis": {
        "description": "An autoimmune disorder causing inflammation in joints.",
        "medicines": ["Methotrexate", "Sulfasalazine", "Hydroxychloroquine", "Adalimumab", "Etanercept"],
        "specialists": ["Rheumatologist", "Orthopedic Specialist", "Physical Therapist"]
    },
    "Congestive Heart Failure": {
        "description": "A chronic condition where the heart doesn't pump blood effectively.",
        "medicines": ["ACE Inhibitors", "Beta Blockers", "Diuretics", "Spironolactone", "Digoxin"],
        "specialists": ["Cardiologist", "General Physician", "Cardiac Surgeon"]
    },
    "Pulmonary Embolism": {
        "description": "A blockage in one of the pulmonary arteries in the lungs.",
        "medicines": ["Heparin", "Warfarin", "Alteplase", "Rivaroxaban", "Dabigatran"],
        "specialists": ["Pulmonologist", "Hematologist", "Emergency Medicine Specialist"]
    },
    "Sepsis": {
        "description": "A life-threatening organ dysfunction caused by a dysregulated immune response to infection.",
        "medicines": ["Vancomycin", "Meropenem", "Piperacillin-Tazobactam", "Cefepime", "Dopamine"],
        "specialists": ["Infectious Disease Specialist", "Intensivist", "Emergency Medicine Specialist"]
    },
    "Liver Cirrhosis": {
        "description": "A late-stage liver disease caused by liver scarring and damage.",
        "medicines": ["Spironolactone", "Furosemide", "Lactulose", "Nadolol", "Rifaximin"],
        "specialists": ["Hepatologist", "Gastroenterologist", "Nutritionist"]
    },
    "Acute Renal Failure": {
        "description": "A sudden loss of kidney function.",
        "medicines": ["Diuretics", "Dopamine", "Calcium Gluconate", "Sodium Bicarbonate", "Epoetin"],
        "specialists": ["Nephrologist", "Critical Care Specialist", "Internal Medicine Specialist"]
    },
    "Urinary Tract Infection": {
        "description": "An infection in any part of the urinary system.",
        "medicines": ["Nitrofurantoin", "Ciprofloxacin", "Amoxicillin-Clavulanate", "Trimethoprim-Sulfamethoxazole", "Cephalexin"],
        "specialists": ["Urologist", "General Physician", "Infectious Disease Specialist"]
    },
    "Hypertension": {
        "description": "A condition in which the force of the blood against the artery walls is too high.",
        "medicines": ["Lisinopril", "Amlodipine", "Losartan", "Hydrochlorothiazide", "Metoprolol"],
        "specialists": ["Cardiologist", "General Physician", "Nephrologist"]
    },
    "Asthma": {
        "description": "A condition in which the airways narrow and swell, causing difficulty in breathing.",
        "medicines": ["Albuterol", "Fluticasone", "Montelukast", "Budesonide", "Salmeterol"],
        "specialists": ["Pulmonologist", "Allergist", "General Physician"]
    },
    "Gastroesophageal Reflux Disease (GERD)": {
        "description": "A digestive disorder where stomach acid irritates the esophagus.",
        "medicines": ["Omeprazole", "Esomeprazole", "Ranitidine", "Lansoprazole", "Pantoprazole"],
        "specialists": ["Gastroenterologist", "General Physician", "Dietitian"]
    }
}



# Extended clean_text function with more steps
def clean_text(text):
    stop_words = set(stopwords.words('english'))

    # Convert to string and lowercase the text
    text = str(text).lower()

    # Remove any numbers (you may want to modify this if numbers are important)
    text = re.sub(r'\d+', '', text)

    # Remove special characters, punctuation, and non-alphabetical characters
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

# Function to make prediction
def predict_disease(patient_note, model, tokenizer, label_encoder):
    patient_note = clean_text(patient_note)

    # Tokenize the input patient note
    inputs = tokenizer(patient_note, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted label
    predicted_label = torch.argmax(logits, dim=1).item()

    # Convert the predicted label to the corresponding disease name
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]

    return predicted_disease


# Route for rendering the index page
@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



# Function to get disease details
def get_disease_details(disease_name):
    if disease_name in disease_data:
        return disease_data[disease_name]
    return {
        "description": "No details available for this disease.",
        "medicines": [],
        "specialists": []
    }

# Updated predict endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    text = ""

    # Extract text from PDF or TXT file
    if file.filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.filename.endswith(".txt"):
        text = content.decode("utf-8")

    # Predict disease
    predicted_disease = predict_disease(text, model, tokenizer, label_encoder)
    disease_details = get_disease_details(predicted_disease)

    # Return result
    return JSONResponse(content={
        "predicted_disease": predicted_disease,
        "description": disease_details["description"],
        "medicines": disease_details["medicines"],
        "specialists": disease_details["specialists"]
    })


# Run the application with Uvicorn
# Command: uvicorn app:app --reload
