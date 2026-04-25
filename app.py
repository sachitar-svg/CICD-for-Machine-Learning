# app.py
# ---------------------------------------------------------------
# FastAPI server — Student Grade Predictor
# All 27 features match exactly what the model was trained on
# ---------------------------------------------------------------

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import os

# ── 1. LOAD MODEL ───────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Model", "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"model.pkl not found at {MODEL_PATH}! Please run train.py first."
    )

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")

# ── 2. FASTAPI APP ───────────────────────────────────────────────
app = FastAPI(
    title="Student Grade Predictor API",
    description="Predicts if a student will PASS or FAIL.",
    version="1.0.0"
)

# ── 3. INPUT SCHEMA — all 27 features ───────────────────────────
class StudentInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "school": 0, "sex": 0, "age": 17,
                "Medu": 3, "Fedu": 2,
                "Mjob": 2, "Fjob": 2, "reason": 1, "guardian": 0,
                "traveltime": 1, "studytime": 2, "failures": 0,
                "schoolsup": 0, "famsup": 1, "paid": 0,
                "activities": 1, "nursery": 1, "higher": 1,
                "internet": 1, "romantic": 0,
                "famrel": 4, "freetime": 3, "goout": 2,
                "Dalc": 1, "Walc": 2, "health": 4, "absences": 4
            }
        }
    )
    school: int       # 0=GP, 1=MS
    sex: int          # 0=F, 1=M
    age: int          # 15–22
    Medu: int         # Mother education (0–4)
    Fedu: int         # Father education (0–4)
    Mjob: int         # Mother job label-encoded (0–4)
    Fjob: int         # Father job label-encoded (0–4)
    reason: int       # Reason to choose school (0–3)
    guardian: int     # 0=mother, 1=father, 2=other
    traveltime: int   # 1=<15min … 4=>1hr
    studytime: int    # 1=<2hrs … 4=>10hrs
    failures: int     # Past failures (0–3)
    schoolsup: int    # School extra support (0=no,1=yes)
    famsup: int       # Family support (0=no,1=yes)
    paid: int         # Paid classes (0=no,1=yes)
    activities: int   # Extracurricular (0=no,1=yes)
    nursery: int      # Attended nursery (0=no,1=yes)
    higher: int       # Wants higher education (0=no,1=yes)
    internet: int     # Internet at home (0=no,1=yes)
    romantic: int     # In relationship (0=no,1=yes)
    famrel: int       # Family relations quality (1–5)
    freetime: int     # Free time after school (1–5)
    goout: int        # Going out with friends (1–5)
    Dalc: int         # Workday alcohol (1–5)
    Walc: int         # Weekend alcohol (1–5)
    health: int       # Health status (1–5)
    absences: int     # School absences (0–93)

# ── 4. ENDPOINTS ─────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "Student Grade Predictor API is running!",
        "docs": "Visit http://localhost:8000/docs to test the API"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(student: StudentInput):
    # Build DataFrame with exact column order from training
    features = pd.DataFrame([[
        student.school, student.sex, student.age,
        student.Medu, student.Fedu,
        student.Mjob, student.Fjob, student.reason, student.guardian,
        student.traveltime, student.studytime, student.failures,
        student.schoolsup, student.famsup, student.paid,
        student.activities, student.nursery, student.higher,
        student.internet, student.romantic,
        student.famrel, student.freetime, student.goout,
        student.Dalc, student.Walc, student.health, student.absences
    ]], columns=[
        'school', 'sex', 'age', 'Medu', 'Fedu',
        'Mjob', 'Fjob', 'reason', 'guardian',
        'traveltime', 'studytime', 'failures',
        'schoolsup', 'famsup', 'paid',
        'activities', 'nursery', 'higher',
        'internet', 'romantic',
        'famrel', 'freetime', 'goout',
        'Dalc', 'Walc', 'health', 'absences'
    ])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    confidence = round(float(max(probability)) * 100, 2)

    result = "PASS ✅" if prediction == 1 else "FAIL ❌"
    advice = (
        "Great job! Keep it up." if prediction == 1
        else "Consider studying more and reducing absences."
    )

    return {
        "prediction": result,
        "confidence": f"{confidence}%",
        "advice": advice,
        "details": {
            "pass_probability": f"{round(probability[1]*100, 2)}%",
            "fail_probability": f"{round(probability[0]*100, 2)}%"
        }
    }