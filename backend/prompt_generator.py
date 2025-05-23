from openai import OpenAI
from typing import List, Optional
import numpy as np
import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()
logger = logging.getLogger(__name__)

class PromptGenerator:
    def __init__(self, openai_client):
        self.openai_client = openai_client

        self.base_system_message_template = """You are a professional image prompt generation expert. Your tasks are:
        1. Generate {num_prompts} different detailed and vivid image prompts based on user's brief description
        2. Each prompt should contain rich visual elements and artistic styles
        3. Ensure the generated prompts are suitable for AI image generation models like Stable Diffusion or DALL·E
        4. Each prompt should be unique, creative and diverse."""
    
    def generate_prompts(
        self,
        user_description: str,
        additional_context: Optional[str] = None,
        num_prompts: int = 10,
        style_preferences: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate image prompts.

        Args:
            user_description: User's input image description
            additional_context: Optional additional context information (such as previous prompts for in-context learning)
            num_prompts: Number of prompts to generate, default is 10
            style_preferences: Optional list of style preferences
        """
        # Build complete system message
        system_message = self.base_system_message_template.format(num_prompts=num_prompts)
        if style_preferences:
            system_message += f"\nPlease focus on these styles: {', '.join(style_preferences)}"
        if additional_context:
            system_message += f"\nAdditional requirements or context: {additional_context}"
            
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Please generate {num_prompts} different image prompts based on this description, each prompt should be unique and creative: {user_description}"}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            # Parse response and return prompt list
            generated_text = response.choices[0].message.content
            prompts = [prompt.strip() for prompt in generated_text.split('\n') if prompt.strip()]
            logger.info(f'generating Prompts: {prompts}')
            return prompts[:num_prompts]
            
        except Exception as e:
            print(f"Error occurred while generating prompts: {str(e)}")
            return []

if __name__ == "__main__":
    from flask import Flask
    import os
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv()
    
    # 创建测试应用
    app = Flask(__name__)
    app.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # 在应用上下文中运行
    with app.app_context():
        # 初始化 generator
        generator = PromptGenerator(app.openai_client)
        
        # 测试用例
        user_input = "一只可爱的猫咪"
        prompts = generator.generate_prompts(
            user_description=user_input,
            num_prompts=2
        )
        
        # 打印结果
        print("\n生成的提示词：")
        for i, prompt in enumerate(prompts, 1):
            print(f"{i}. {prompt}") 