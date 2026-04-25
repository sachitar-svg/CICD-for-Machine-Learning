from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import joblib, pandas as pd, os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "Model", "model.pkl")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Student Grade Predictor API")

class StudentInput(BaseModel):
    model_config = ConfigDict()
    school:int; sex:int; age:int; Medu:int; Fedu:int
    Mjob:int; Fjob:int; reason:int; guardian:int
    traveltime:int; studytime:int; failures:int
    schoolsup:int; famsup:int; paid:int
    activities:int; nursery:int; higher:int
    internet:int; romantic:int
    famrel:int; freetime:int; goout:int
    Dalc:int; Walc:int; health:int; absences:int

@app.get("/")
def home(): return {"message": "Student Grade Predictor API is running!"}

@app.get("/health")
def health(): return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(s: StudentInput):
    features = pd.DataFrame([[
        s.school,s.sex,s.age,s.Medu,s.Fedu,
        s.Mjob,s.Fjob,s.reason,s.guardian,
        s.traveltime,s.studytime,s.failures,
        s.schoolsup,s.famsup,s.paid,
        s.activities,s.nursery,s.higher,
        s.internet,s.romantic,
        s.famrel,s.freetime,s.goout,
        s.Dalc,s.Walc,s.health,s.absences
    ]], columns=[
        'school','sex','age','Medu','Fedu',
        'Mjob','Fjob','reason','guardian',
        'traveltime','studytime','failures',
        'schoolsup','famsup','paid',
        'activities','nursery','higher',
        'internet','romantic',
        'famrel','freetime','goout',
        'Dalc','Walc','health','absences'
    ])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    return {
        "prediction": "PASS ✅" if pred==1 else "FAIL ❌",
        "confidence": f"{round(float(max(prob))*100,2)}%",
        "advice": "Keep it up!" if pred==1 else "Study more and reduce absences.",
        "details": {
            "pass_probability": f"{round(prob[1]*100,2)}%",
            "fail_probability": f"{round(prob[0]*100,2)}%"
        }
    }