import gradio as gr
from skops.io import load, get_untrusted_types

trusted_types = get_untrusted_types(file="Model/drug_pipeline.skops")
pipe = load(file="Model/drug_pipeline.skops", trusted=trusted_types)

def predict_drug(age, sex, bp, chol, na_k):
    return f"Predicted Drug: {pipe.predict([[age, sex, bp, chol, na_k]])[0]}"

gr.Interface(
    fn=predict_drug,
    inputs=[
        gr.Slider(15, 74, label="Age"),
        gr.Radio(["M", "F"], label="Sex"),
        gr.Radio(["HIGH", "LOW", "NORMAL"], label="BP"),
        gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
        gr.Slider(6.2, 38.2, label="Na_to_K")
    ],
    outputs=gr.Label(),
    examples=[[30, "M", "HIGH", "NORMAL", 15.4]],
    title="Drug Classification",
    description="Predict drug class based on patient features"
).launch()
