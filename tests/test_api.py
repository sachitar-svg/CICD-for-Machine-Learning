from fastapi.testclient import TestClient
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app
client = TestClient(app)

BASE = {
    "school":0,"sex":0,"age":17,"Medu":3,"Fedu":2,
    "Mjob":2,"Fjob":2,"reason":1,"guardian":0,
    "traveltime":1,"studytime":2,"failures":0,
    "schoolsup":0,"famsup":1,"paid":0,
    "activities":1,"nursery":1,"higher":1,
    "internet":1,"romantic":0,
    "famrel":4,"freetime":3,"goout":2,
    "Dalc":1,"Walc":2,"health":4,"absences":4
}

def test_home():
    r = client.get("/")
    assert r.status_code == 200

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict_good_student():
    r = client.post("/predict", json={**BASE, "studytime":3, "failures":0})
    assert r.status_code == 200
    assert "prediction" in r.json()

def test_predict_at_risk_student():
    r = client.post("/predict", json={**BASE, "studytime":1, "failures":3, "absences":25})
    assert r.status_code == 200
    assert "prediction" in r.json()

def test_missing_field():
    r = client.post("/predict", json={"age": 17})
    assert r.status_code == 422