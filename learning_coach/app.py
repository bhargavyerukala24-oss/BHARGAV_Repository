import os
import streamlit as st
from core import StreamResult, context_messages, openai_stream, ollama_stream

st.set_page_config(page_title='Learning Coach', page_icon='📘')
st.title('📘 Python & GenAI Learning Coach')
st.caption('A teaching assistant with streamed answers and session history.')
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'usage' not in st.session_state:
    st.session_state.usage = []

with st.sidebar:
    st.header('Choose your model')
    backend = st.radio('Provider', ['Ollama (local)', 'OpenAI API'])
    level = st.selectbox('Learning level', ['Beginner', 'Intermediate'])
    if backend == 'OpenAI API':
        model = st.text_input('OpenAI model', 'gpt-4.1-mini')
        try:
            secret = st.secrets.get('OPENAI_API_KEY', '')
        except FileNotFoundError:
            secret = ''
        api_key = os.environ.get('OPENAI_API_KEY') or secret
        st.caption('API use may be billed separately from ChatGPT.')
    else:
        model = st.text_input('Local model', 'qwen2.5-coder:1.5b')
        st.caption('Ollama must be running on this computer.')
    if st.button('Clear conversation'):
        st.session_state.messages = []
        st.session_state.usage = []
        st.rerun()
    st.caption('History stays in this browser session. The most recent 6 turns are sent with each question.')
    if st.session_state.usage:
        last = st.session_state.usage[-1]
        st.subheader('Last completed request')
        st.write(f"Provider: {last['provider']} | Model: {last['model']}")
        st.write(f"Input tokens: {last['input'] if last['input'] is not None else 'not reported'}")
        st.write(f"Output tokens: {last['output'] if last['output'] is not None else 'not reported'}")
        st.write(f"Total tokens: {last['total'] if last['total'] is not None else 'not reported'}")
        known = [x for x in st.session_state.usage if x['total'] is not None]
        if known:
            st.caption(f"Reported session total: {sum(x['total'] for x in known)} tokens across {len(known)} requests with reported totals. Providers count differently.")
    st.caption('Token counts are not exact cost. Check your provider usage dashboard.')

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

question = st.chat_input('Ask about Python or Generative AI', max_chars=1200)
if question:
    if backend == 'OpenAI API' and not api_key:
        st.error('Configure OPENAI_API_KEY before choosing OpenAI API. See README.md.')
        st.stop()
    instructions = (f'You are a Python and Generative AI teacher for a {level} learner. '
        'Explain using simple English, one short example and one practice question. '
        'For code, explain key lines. Be honest about uncertainty. '
        'If the user asks for a hint, give a hint before the full answer.')
    messages = context_messages(st.session_state.messages, question)
    with st.chat_message('user'):
        st.markdown(question)
    result = StreamResult()
    try:
        if backend == 'OpenAI API':
            from openai import OpenAI
            client = OpenAI(api_key=api_key, timeout=45, max_retries=2)
            stream = openai_stream(client, model, instructions, messages, result)
        else:
            import requests
            stream = ollama_stream(requests, 'http://127.0.0.1:11434', model,
                                   instructions, messages, result)
        with st.chat_message('assistant'):
            answer = st.write_stream(stream)
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError('No text returned; try another question or model.')
        st.session_state.messages.extend([
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer},
        ])
        u = result.usage
        st.session_state.usage.append({'provider': backend, 'model': model,
            'input': u.input_tokens if u else None,
            'output': u.output_tokens if u else None,
            'total': u.total_tokens if u else None})
        if result.status == 'incomplete':
            st.session_state.incomplete_notice = True
        st.rerun()
    except Exception as exc:
        st.error(f'Request failed ({type(exc).__name__}). Check the provider, model, network, API credits, or Ollama service. The failed turn was not saved.')
if st.session_state.pop('incomplete_notice', False):
    st.warning('The last answer reached a limit and may be incomplete.')
