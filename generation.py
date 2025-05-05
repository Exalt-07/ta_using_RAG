from openai import OpenAI
import yaml
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ResponseGenerator:
    def __init__(self):
        # Initialize OpenAI client with API key from environment
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load prompt templates
        with open('app/config/prompts.yaml') as f:
            self.prompts = yaml.safe_load(f)

    def generate(self, context, query, student_level='intermediate', images=None):
        prompt_template = self.prompts[student_level]
        
        # Prepare image context if images are provided
        image_context = ""
        if images:
            image_context = "\nRelevant diagrams: " + ", ".join([str(img) for img in images])
        
        # Support {images} in prompt templates, fallback if not present
        try:
            formatted_prompt = prompt_template.format(
                context=context,
                question=query,
                images=image_context
            )
        except KeyError:
            # If the prompt template does not have {images}
            formatted_prompt = prompt_template.format(
                context=context,
                question=query
            ) + image_context
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "You are a patient teaching assistant. Use only the provided context."
            }, {
                "role": "user", 
                "content": formatted_prompt
            }]
        )
        return response.choices[0].message.content
