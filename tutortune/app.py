from pathlib import Path
import streamlit as st
from inference import Tutor

st.set_page_config(page_title="TutorTune")
st.title("TutorTune - base versus LoRA adapter")
st.caption("A small Python teaching experiment; not a validated tutor.")
if not Path("adapter/training_metadata.json").exists():
    st.info("Train first, then place the complete adapter folder here.")
    st.stop()


@st.cache_resource
def load_tutor():
    return Tutor()


mode = st.radio("Model", ["Base model", "LoRA adapter"])
question = st.text_area(
    "Python question", "Explain Python functions.", max_chars=700
)
if st.button("Generate", disabled=not question.strip()):
    try:
        with st.spinner("Loading / generating..."):
            answer = load_tutor().answer(
                question, use_adapter=mode == "LoRA adapter"
            )
        st.write(answer)
    except Exception as exc:
        st.error(
            f"Generation failed ({type(exc).__name__}). "
            "Check adapter files, base download and available memory."
        )
