# Install necessary packages
!pip install gradio
!pip install transformers torch accelerate
!huggingface-cli login

!huggingface-cli whoami

# Import required libraries
!pip install httpx==0.24.1

import gradio as gr
from transformers import pipeline

# Function to generate a blog based on user inputs
def generate_blog(input_text, no_words, model_name):
    generator = pipeline("text-generation", model=model_name)

    prompt = f"Write a blog for {model_name} job profile for a topic {input_text} within {no_words} words:"
    response = generator(prompt, max_length=int(no_words), do_sample=True, temperature=0.7)[0]['generated_text']
    return response
## Define input components for the Gradio interface
inputs = [
    gr.Textbox(lines=5, label="Enter the Blog Topic"),
    gr.Textbox(lines=1, label="No of Words"),
    gr.Dropdown(choices=[
        'meta-llama/Llama-2-7b-chat',  # Fully qualified model name
        'EleutherAI/gpt-neo-2.7B',  # Corrected model name
        'facebook/bart-base'  # Potential alternative model
    ], label="Choose a Model", value="meta-llama/Llama-2-7b-chat")  # Set initial value
]

# Define output component for the Gradio interface
output = gr.Textbox(label="Blog Generation")
# Create Gradio interface with the defined input and output components
interface = gr.Interface(generate_blog, inputs, output, title="Generate Blogs")

# For Colab notebooks, ensure sharing is enabled and set debug if needed
# Launch the Gradio interface, allowing sharing and enabling debug mode
interface.launch(share=True, debug=True)
