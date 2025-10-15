# Import necessary packages
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

# Model and project settings
model_id = "meta-llama/llama-3-2-11b-vision-instruct"  # LLAMA3 model

# Set credentials (Cloud IDE has built-in access)
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
)

# Model configuration
params = TextChatParameters()

# Project ID for Skills Network (default free access)
project_id = "skills-network"

# Initialize the model
model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params
)

# Define your question
prompt_txt = "How to be a good Data Scientist?"

# Create conversation messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_txt},
        ]
    }
]

# Generate a response
generated_response = model.chat(messages=messages)
generated_text = generated_response['choices'][0]['message']['content']

# Print the model's response
print(generated_text)
