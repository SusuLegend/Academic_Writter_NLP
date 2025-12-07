import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# ================================
# 🔹 Load LoRA Academic Model
# ================================
BASE_MODEL = "EleutherAI/gpt-neo-125M"
LORA_MODEL = "Lora_Academic_Model"   # <-- change to your path or HF repo

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base, LORA_MODEL)
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True
    )
    return pipe

pipe = load_model()

# ================================
# 🔹 Streamlit UI
# ================================
st.set_page_config(
    page_title="Academic Text Generator",
    layout="wide",
)

st.title("📘 Academic Next-Text Generator")
st.markdown("""
A clean interface for generating structured academic writing using your LoRA fine-tuned model.
""")

# Sidebar for model parameters
with st.sidebar:
    st.header("⚙️ Model Settings")

    max_len = st.slider("Max Output Tokens", 50, 1500, 300, 50)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
    top_p = st.slider("Top-P", 0.1, 1.0, 0.9, 0.05)

# Main input area
st.subheader("🔍 Input Academic Prompt")
prompt = st.text_area(
    "",
    placeholder="Example: Discuss the impact of model scaling on generalization in NLP...",
    height=180
)

# Generate text
if st.button("Generate Academic Text"):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Generating academic text…"):
            out = pipe(
                prompt,
                max_new_tokens=max_len,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=pipe.tokenizer.eos_token_id
            )[0]["generated_text"]

        st.subheader("📄 Generated Academic Text")
        st.markdown(
            f"""
            <div style="background:#F7F7F9;padding:15px;border-radius:8px;">
            {out}
            </div>
            """,
            unsafe_allow_html=True
        )
