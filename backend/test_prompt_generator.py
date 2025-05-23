from openai import OpenAI
from prompt_generator import PromptGenerator
import os
from dotenv import load_dotenv
import logging
from flask import Flask

# 设置日志
logging.basicConfig(level=logging.INFO)

# 加载环境变量
load_dotenv()

def create_test_app():
    app = Flask(__name__)
    app.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    return app

def test_prompt_generator():
    # 创建测试应用
    app = create_test_app()
    
    # 在应用上下文中运行
    with app.app_context():
        # 初始化 PromptGenerator
        generator = PromptGenerator(app.openai_client)
        
        # 测试用例
        test_description = "一只可爱的猫咪"
        
        # 生成提示词
        prompts = generator.generate_prompts(
            user_description=test_description,
            num_prompts=2  # 生成2个提示词用于测试
        )
        
        # 打印结果
        print("\n生成的提示词：")
        for i, prompt in enumerate(prompts, 1):
            print(f"{i}. {prompt}")

if __name__ == "__main__":
    test_prompt_generator() 