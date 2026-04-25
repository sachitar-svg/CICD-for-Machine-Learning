# app_gradio.py
# ---------------------------------------------------------------
# Gradio web UI for the Student Grade Predictor.
# This is what gets deployed to Hugging Face Spaces.
# Users can fill in a form and get predictions instantly.
# ---------------------------------------------------------------

import gradio as gr
import joblib
import pandas as pd
import os

# ── 1. LOAD MODEL ───────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Model", "model.pkl")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded!")
print("🚀 Starting Gradio app...")
# ── 2. PREDICTION FUNCTION ───────────────────────────────────────
# Gradio calls this function every time the user clicks Predict.
# It receives values directly from the UI sliders/dropdowns.

def predict_grade(
    school, sex, age,
    Medu, Fedu, Mjob, Fjob, reason, guardian,
    traveltime, studytime, failures,
    schoolsup, famsup, paid, activities, nursery, higher,
    internet, romantic,
    famrel, freetime, goout, Dalc, Walc, health, absences
):
    # Map human-readable labels back to numbers
    school_map   = {"GP": 0, "MS": 1}
    sex_map      = {"Female": 0, "Male": 1}
    job_map      = {"at_home": 0, "health": 1, "other": 2, "services": 3, "teacher": 4}
    reason_map   = {"course": 0, "home": 1, "other": 2, "reputation": 3}
    guardian_map = {"mother": 0, "father": 1, "other": 2}
    yn_map       = {"No": 0, "Yes": 1}

    features = pd.DataFrame([[
        school_map[school],
        sex_map[sex],
        age,
        Medu, Fedu,
        job_map[Mjob],
        job_map[Fjob],
        reason_map[reason],
        guardian_map[guardian],
        traveltime, studytime, failures,
        yn_map[schoolsup],
        yn_map[famsup],
        yn_map[paid],
        yn_map[activities],
        yn_map[nursery],
        yn_map[higher],
        yn_map[internet],
        yn_map[romantic],
        famrel, freetime, goout,
        Dalc, Walc, health, absences
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

    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    pass_prob   = round(probability[1] * 100, 1)
    fail_prob   = round(probability[0] * 100, 1)

    if prediction == 1:
        result = f"✅ PASS  ({pass_prob}% confidence)"
        advice = "🎉 Great work! Keep studying consistently and you're on track."
    else:
        result = f"❌ FAIL  ({fail_prob}% confidence)"
        advice = "📚 Try increasing your study time and reducing absences. You can do it!"

    breakdown = (
        f"Pass probability : {pass_prob}%\n"
        f"Fail probability : {fail_prob}%"
    )

    return result, advice, breakdown


# ── 3. BUILD GRADIO UI ───────────────────────────────────────────
# gr.Blocks lets us design a custom layout with sections.

with gr.Blocks(title="Student Grade Predictor", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🎓 Student Grade Predictor
    ### Will the student PASS or FAIL their final exam?
    Fill in the student details below and click **Predict** to find out.
    > Model trained on UCI Student Performance Dataset (Math course)
    """)

    with gr.Row():

        # ── Left column: Student background ─────────────────────
        with gr.Column():
            gr.Markdown("### 👤 Student Background")

            school = gr.Dropdown(
                ["GP", "MS"], label="School",
                info="GP = Gabriel Pereira | MS = Mousinho da Silveira",
                value="GP"
            )
            sex = gr.Dropdown(
                ["Female", "Male"], label="Sex", value="Female"
            )
            age = gr.Slider(
                15, 22, value=17, step=1, label="Age"
            )
            guardian = gr.Dropdown(
                ["mother", "father", "other"],
                label="Guardian", value="mother"
            )
            internet = gr.Dropdown(
                ["Yes", "No"], label="Internet access at home?", value="Yes"
            )
            romantic = gr.Dropdown(
                ["No", "Yes"], label="In a romantic relationship?", value="No"
            )

        # ── Middle column: Academic info ─────────────────────────
        with gr.Column():
            gr.Markdown("### 📚 Academic Details")

            studytime = gr.Slider(
                1, 4, value=2, step=1,
                label="Weekly study time",
                info="1=<2hrs | 2=2-5hrs | 3=5-10hrs | 4=>10hrs"
            )
            failures = gr.Slider(
                0, 3, value=0, step=1,
                label="Number of past failures"
            )
            absences = gr.Slider(
                0, 93, value=4, step=1,
                label="Number of school absences"
            )
            higher = gr.Dropdown(
                ["Yes", "No"],
                label="Wants to pursue higher education?", value="Yes"
            )
            schoolsup = gr.Dropdown(
                ["No", "Yes"], label="Extra school support?", value="No"
            )
            paid = gr.Dropdown(
                ["No", "Yes"], label="Extra paid classes?", value="No"
            )
            reason = gr.Dropdown(
                ["course", "home", "reputation", "other"],
                label="Reason for choosing school", value="course"
            )

        # ── Right column: Family & lifestyle ────────────────────
        with gr.Column():
            gr.Markdown("### 🏠 Family & Lifestyle")

            Medu = gr.Slider(
                0, 4, value=2, step=1,
                label="Mother's education",
                info="0=none | 1=primary | 2=5th-9th | 3=secondary | 4=higher"
            )
            Fedu = gr.Slider(
                0, 4, value=2, step=1,
                label="Father's education"
            )
            Mjob = gr.Dropdown(
                ["at_home", "health", "other", "services", "teacher"],
                label="Mother's job", value="other"
            )
            Fjob = gr.Dropdown(
                ["at_home", "health", "other", "services", "teacher"],
                label="Father's job", value="other"
            )
            famrel = gr.Slider(
                1, 5, value=4, step=1,
                label="Family relationship quality (1=bad, 5=excellent)"
            )
            famsup = gr.Dropdown(
                ["Yes", "No"], label="Family educational support?", value="Yes"
            )
            nursery = gr.Dropdown(
                ["Yes", "No"], label="Attended nursery school?", value="Yes"
            )
            activities = gr.Dropdown(
                ["No", "Yes"], label="Extracurricular activities?", value="No"
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎉 Social & Health")
            traveltime = gr.Slider(
                1, 4, value=1, step=1,
                label="Travel time to school",
                info="1=<15min | 2=15-30min | 3=30min-1hr | 4=>1hr"
            )
            freetime = gr.Slider(
                1, 5, value=3, step=1,
                label="Free time after school (1=very low, 5=very high)"
            )
            goout = gr.Slider(
                1, 5, value=3, step=1,
                label="Going out with friends (1=very low, 5=very high)"
            )
            health = gr.Slider(
                1, 5, value=4, step=1,
                label="Current health status (1=very bad, 5=very good)"
            )
            Dalc = gr.Slider(
                1, 5, value=1, step=1,
                label="Workday alcohol consumption (1=very low, 5=very high)"
            )
            Walc = gr.Slider(
                1, 5, value=1, step=1,
                label="Weekend alcohol consumption (1=very low, 5=very high)"
            )

    # ── Predict button ───────────────────────────────────────────
    predict_btn = gr.Button("🔮 Predict", variant="primary", size="lg")

    # ── Output section ───────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### 📊 Prediction Result")

    with gr.Row():
        result_out    = gr.Textbox(label="Prediction", interactive=False)
        advice_out    = gr.Textbox(label="Advice", interactive=False)
        breakdown_out = gr.Textbox(label="Probability Breakdown", interactive=False)

    # ── Wire button to function ──────────────────────────────────
    predict_btn.click(
        fn=predict_grade,
        inputs=[
            school, sex, age,
            Medu, Fedu, Mjob, Fjob, reason, guardian,
            traveltime, studytime, failures,
            schoolsup, famsup, paid, activities, nursery, higher,
            internet, romantic,
            famrel, freetime, goout, Dalc, Walc, health, absences
        ],
        outputs=[result_out, advice_out, breakdown_out]
    )

    gr.Markdown("""
    ---
    *Built with Scikit-learn + Gradio · Deployed via GitHub Actions · Part of MLOps mini project*
    """)

# ── 4. LAUNCH ────────────────────────────────────────────────────
if __name__ == "__main__":
    # share=True gives you a public link (works without Hugging Face too!)
    demo.launch(share=True)