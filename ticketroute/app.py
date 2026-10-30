import os
import streamlit as st
from classifier import load_model, classify
from config import LABELS

st.set_page_config(page_title='TicketRoute', page_icon='🎫')
st.title('🎫 TicketRoute')
st.caption('Suggest a support team and flag uncertain tickets for review.')
with st.sidebar:
    mode = st.radio('Model', ['Zero-shot baseline', 'Fine-tuned classifier'])
    threshold = st.slider('Minimum top score', 0.0, 1.0, 0.65, 0.05)
    margin = st.slider('Minimum score gap', 0.0, 0.5, 0.15, 0.05)
    st.caption('Defaults are starting points, not calibrated probabilities. Tune on validation examples.')
    show_tokens = st.checkbox('Show tokenizer example', value=True)
    st.write('Supported categories:', ', '.join(LABELS))

@st.cache_resource
def cached_model(selected_mode): return load_model(selected_mode)

text = st.text_area('Describe the support issue', max_chars=1500,
                    placeholder='I was charged twice for my subscription.')
if st.button('Classify ticket', type='primary', disabled=not text.strip()):
    try:
        if mode == 'Fine-tuned classifier':
            # Load root-level Streamlit secrets before reading environment variables.
            try:
                configured_id = st.secrets.get('TICKET_MODEL_ID', '')
                if configured_id:
                    os.environ.setdefault('TICKET_MODEL_ID', configured_id)
            except FileNotFoundError:
                pass
        with st.spinner('Loading model or classifying...'):
            model = cached_model(mode)
            result = classify(model, mode, text, threshold, margin)
        if result['review']:
            st.warning('Manual review recommended. Suggested category: ' + result['label'])
        else: st.success('Suggested category: ' + result['label'])
        st.caption(f"Top score: {result['score']:.3f}; gap to second: {result['margin']:.3f}")
        st.table([{'category': label, 'score': round(score, 3)}
                  for label, score in sorted(result['scores'].items(),
                                             key=lambda item: item[1], reverse=True)])
        st.caption('Scores are not calibrated correctness probabilities. This tool does not create or route real tickets.')
        if show_tokens:
            with st.expander('See how the tokenizer represents this ticket'):
                encoded = model.tokenizer(text, truncation=True, max_length=256)
                st.write('Token strings:', model.tokenizer.convert_ids_to_tokens(encoded['input_ids']))
                st.write('Token IDs:', encoded['input_ids'])
                st.caption('This illustrates ticket tokenization alone. Zero-shot also tokenizes label hypotheses.')
    except Exception as exc:
        st.error(f'Could not load or run the model ({type(exc).__name__}). Check model download, memory, package versions and README.')
