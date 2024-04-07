import streamlit as st
from threading import Thread

st.set_page_config(
    page_title="creek model 演示(暂未训练多轮对话)",
    page_icon=":robot:",
    layout='wide'
)
device = "cuda:0"
st.title("Creek")

@st.cache_resource
def get_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    class FastapiTaskStramer(TextIteratorStreamer):
        def __init__(self, tokenizer, skip_prompt = False, timeout= None, **decode_kwargs
            ):
            super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)
        def end(self):
            if len(self.token_cache) > 0:
                text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
                printable_text = text[self.print_len :]
                self.print_len = 0
            else:
                printable_text = ""
            self.next_tokens_are_prompt = True
            self.on_finalized_text(printable_text, stream_end=False)
            # self.on_finalized_text(printable_text, stream_end=True)
    model_path = "maheer/creek"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model = model.eval()
    streamer = FastapiTaskStramer(tokenizer, skip_prompt=True, timeout=180)
    messages = [
            {'role':'user','content':"?"},
    ]
    return tokenizer, model, streamer, messages


tokenizer, model, streamer, messages = get_model()


def generate_stream(**kwargs):
    thread = Thread( target=model.generate, kwargs = kwargs )
    thread.start()
    for new_text in enumerate(streamer):
        if tokenizer.eos_token in new_text:
            print(new_text)
        if new_text:
            yield new_text

def generate(**kwargs):
    output = model.generate(**kwargs)
    intput_len = len(kwargs['input_ids'][0])
    output = output[0][intput_len:]
    return tokenizer.decode(output,skip_special_tokens=True)

max_length = st.sidebar.slider(
    'max_length', 0, 1024, 1024, step=1
)
top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.8, step=0.01
)
temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.3, step=0.01
)
repetition_penalty = st.sidebar.slider(
    'repetition_penalty', 0.0, 4.0, 1.25, step=0.01
)

if 'history' not in st.session_state:
    st.session_state.history = []

if 'past_key_values' not in st.session_state:
    st.session_state.past_key_values = None

for i, (query, response) in enumerate(st.session_state.history):
    with st.chat_message(name="user", avatar="user"):
        st.markdown(query)
    with st.chat_message(name="assistant", avatar="assistant"):
        st.markdown(response)
with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

prompt_text = st.text_area(label="用户命令输入",
                           height=100,
                           placeholder="请在这儿输入您的命令",
                           value="苹果手机怎样连接互联网")

button = st.button("发送", key="predict")

if button:
    input_placeholder.markdown(prompt_text)
    messages[0]['content'] = prompt_text
    inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True)
    if len(inputs["input_ids"][0]) >= max_length :
        message_placeholder.markdown("您好，您输入的问题太长了，已经超过了max_length，请调大max_length，或者精简问题，并重新提交。")
    else:
        history, past_key_values = st.session_state.history, st.session_state.past_key_values
        inputs = inputs.to(device)
        response = generate(
            max_length=max_length, 
            top_p=top_p,
            temperature=temperature,
            do_sample=True, 
            repetition_penalty=repetition_penalty,
            **inputs)
        message_placeholder.markdown(response)
        # for response in generate_stream(max_length=max_length, top_p=top_p, temperature=temperature,do_sample=True, **inputs):
        #     message_placeholder.markdown(response)
    # st.session_state.history = history
    # st.session_state.past_key_values = past_key_values