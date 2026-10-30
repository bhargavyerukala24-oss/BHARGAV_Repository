import os
import streamlit as st
from workflow import TOPICS, load_sources, evidence_workflow
from local_agent import run_agent

st.set_page_config(page_title='CarePrep', page_icon='📚')
st.title('📚 CarePrep — Health Information and Visit Preparation')
st.info('Educational prototype using a small curated set of sources. '
        'It does not diagnose, interpret your test results, prescribe, or assess emergencies. '
        'For urgent concerns, seek local medical help. Do not enter personal medical information.')
st.caption('Sources are static summaries checked on the dates shown, not live medical advice.')

try:
    records = load_sources()
except Exception:
    st.error('Source file is invalid. The demonstration cannot display health summaries.')
    st.stop()

with st.sidebar:
    ai_setting = os.environ.get('ENABLE_LOCAL_AI')
    if ai_setting is None:
        try: ai_setting = str(st.secrets.get('ENABLE_LOCAL_AI', 'true'))
        except FileNotFoundError: ai_setting = 'true'
    modes = ['Evidence and questions']
    if ai_setting.lower() == 'true': modes.append('Optional local AI explanation')
    mode = st.radio('Mode', modes)
    if mode == 'Optional local AI explanation':
        model = st.text_input('Ollama tool-capable model', 'qwen2.5:3b')
        st.caption('AI-generated text is experimental and not clinically validated.')
    st.caption('This version accepts preset general topics only; there is no patient-record upload.')

options = ['Choose a topic'] + [v['label'] for v in TOPICS.values()] + ['Another topic']
choice = st.selectbox('Choose a general topic', options)
if choice == 'Choose a topic': st.stop()
if choice == 'Another topic':
    st.warning('This topic is not covered by the sample sources. Consult an appropriate clinician or authoritative resource.')
    st.stop()
topic = next(k for k, v in TOPICS.items() if v['label'] == choice)
try:
    result = evidence_workflow(topic, records)
except Exception:
    st.error('Could not check the selected source. No health summary will be shown.')
    st.stop()
if result['source']['status'] != 'ok':
    st.warning('The source is missing or due for review. Check the original source and consult a clinician; the summary is withheld.')
    st.stop()

st.subheader('Source-backed summary')
for source in result['source']['sources']:
    st.write(source['summary'] + ' [' + source['id'] + ']')
    st.markdown(f"[{source['id']}: {source['title']}]({source['url']})")
    st.caption(f"{source['publisher']} · {source['source_date_label']}: {source['source_date']} · "
               f"Sample checked: {source['checked_on']} · Not checked live")
st.subheader('Questions you can take to a clinician')
for question in result['visit']['questions']:
    st.write('• ' + question)
st.caption('These are general question templates, not personalized advice.')

if mode == 'Optional local AI explanation':
    st.warning('The AI draft is experimental. A valid citation label does not prove the text is accurate or supported.')
    if st.button('Generate a draft from this source'):
        try:
            with st.spinner('Calling local model and tools...'):
                output = run_agent(model, topic, records)
            if output['answer']:
                st.subheader('Experimental AI draft — review against the source')
                st.write(output['answer'])
            else:
                st.warning('No draft shown: ' + str(output['reason']))
            pass
        except Exception as exc:
            st.error(f'Local model unavailable ({type(exc).__name__}). Use the source-backed summary above.')
