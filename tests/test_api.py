# tests/test_api.py
from fastapi.testclient import TestClient
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

client = TestClient(app)

# Sample with all 27 features
GOOD_STUDENT = {
    "school": 0, "sex": 0, "age": 16,
    "Medu": 4, "Fedu": 4,
    "Mjob": 3, "Fjob": 3, "reason": 1, "guardian": 0,
    "traveltime": 1, "studytime": 3, "failures": 0,
    "schoolsup": 0, "famsup": 1, "paid": 1,
    "activities": 1, "nursery": 1, "higher": 1,
    "internet": 1, "romantic": 0,
    "famrel": 5, "freetime": 2, "goout": 1,
    "Dalc": 1, "Walc": 1, "health": 5, "absences": 2
}

AT_RISK_STUDENT = {
    "school": 1, "sex": 1, "age": 18,
    "Medu": 0, "Fedu": 0,
    "Mjob": 0, "Fjob": 0, "reason": 0, "guardian": 2,
    "traveltime": 4, "studytime": 1, "failures": 3,
    "schoolsup": 0, "famsup": 0, "paid": 0,
    "activities": 0, "nursery": 0, "higher": 0,
    "internet": 0, "romantic": 1,
    "famrel": 1, "freetime": 5, "goout": 5,
    "Dalc": 4, "Walc": 5, "health": 2, "absences": 25
}

def test_home():
    r = client.get("/")
    assert r.status_code == 200
    assert "Student Grade Predictor" in r.json()["message"]

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict_good_student():
    r = client.post("/predict", json=GOOD_STUDENT)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "advice" in data

def test_predict_at_risk_student():
    r = client.post("/predict", json=AT_RISK_STUDENT)
    assert r.status_code == 200
    assert "prediction" in r.json()

def test_missing_field():
    r = client.post("/predict", json={"age": 17})
    assert r.status_code == 422  # FastAPI auto-rejects incomplete input