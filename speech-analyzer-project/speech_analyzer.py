import torch
import gradio as gr
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

#######------------- LLM-------------####
# IBM Watsonx LLM setup
my_credentials = {"url": "https://us-south.ml.cloud.ibm.com"}
params = {
    GenParams.MAX_NEW_TOKENS: 700,
    GenParams.TEMPERATURE: 0.1
}
LLAMA3_model = Model(
    model_id='meta-llama/llama-3-2-11b-vision-instruct',
    credentials=my_credentials,
    params=params,
    project_id="skills-network"
)
llm = WatsonxLLM(LLAMA3_model)

#######------------- Prompt Template-------------####
temp = """
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
"""
pt = PromptTemplate(input_variables=["context"], template=temp)
prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)

#######------------- Speech2text-------------####
def transcript_audio(audio_file):
    # Initialize the speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30
    )
    
    # Transcribe the audio file
    transcript_txt = pipe(audio_file, batch_size=8)["text"]
    
    # Merge transcript with prompt template and send to LLM
    result = prompt_to_LLAMA2.run(transcript_txt) 
    return result

#######------------- Gradio-------------####
audio_input = gr.Audio(type="filepath", label="Upload Audio")  # Corrected for Gradio 5.x
output_text = gr.Textbox(label="Key Points Summary")

# Create the Gradio interface
iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="Speech Analyzer App",
    description="Upload a meeting or lecture audio file to get key points summarized by AI."
)

iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
