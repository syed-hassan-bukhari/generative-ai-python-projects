# Import necessary packages
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
import gradio as gr

# Set credentials
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
)

# Model and project settings
model_id = "meta-llama/llama-3-2-11b-vision-instruct"
project_id = "skills-network"

# Parameters (you can tune these)
params = TextChatParameters(
    temperature=0.2,
    max_tokens=800  # increase this value
)


# Initialize model
model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params
)

# Function to get response
def generate_response(prompt_txt):
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_txt}],
        }
    ]
    generated_response = model.chat(messages=messages)
    generated_text = generated_response['choices'][0]['message']['content']
    return generated_text

# Build Gradio app
chat_application = gr.Interface(
    fn=generate_response,
    flagging_mode="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Watsonx.ai Chatbot",
    description="Ask any question and the chatbot will respond using LLaMA 3 model."
)

# Launch app
chat_application.launch(share=True)
