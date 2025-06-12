import requests
import json
import sys
import os
from dotenv import load_dotenv
from api.config import load_embedder_config, load_generator_config
from adalflow.core.types import ModelType

# 导入QwenClient
from api.qwen_client import QwenClient

# 加载环境变量
def load_env():
    """加载环境变量，优先从项目根目录的.env文件加载"""
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录
    project_root = os.path.dirname(current_dir)
    
    # 尝试从项目根目录加载.env文件
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        print(f"Loading environment variables from {env_path}")
        load_dotenv(env_path)
    else:
        print(f"Warning: .env file not found at {env_path}")
        # 创建一个示例.env文件
        print("Creating example .env file...")
        with open(env_path, 'w') as f:
            f.write("# OpenAI API配置\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
            f.write("OPENAI_API_BASE_URL=https://api.openai.com/v1\n\n")
            f.write("# Qwen API配置\n")
            f.write("QWEN_API_KEY=your_qwen_api_key_here\n")
            f.write("QWEN_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1\n")
        print(f"Example .env file created at {env_path}")
        print("Please edit the .env file with your API keys and try again.")
        sys.exit(1)

def test_streaming_endpoint(repo_url, query, file_path=None):
    """
    Test the streaming endpoint with a given repository URL and query.
    
    Args:
        repo_url (str): The GitHub repository URL
        query (str): The query to send
        file_path (str, optional): Path to a file in the repository
    """
    # Define the API endpoint
    url = "http://localhost:8000/chat/completions/stream"
    
    # Define the request payload
    payload = {
        "repo_url": repo_url,
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "filePath": file_path
    }
    
    print(f"Testing streaming endpoint with:")
    print(f"  Repository: {repo_url}")
    print(f"  Query: {query}")
    if file_path:
        print(f"  File Path: {file_path}")
    print("\nResponse:")
    
    try:
        # Make the request with streaming enabled
        response = requests.post(url, json=payload, stream=True)
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            try:
                error_data = json.loads(response.content)
                print(f"Error details: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"Error content: {response.content}")
            return
        
        # Process the streaming response
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                print(chunk.decode('utf-8'), end='', flush=True)
        
        print("\n\nStreaming completed successfully.")
    
    except Exception as e:
        print(f"Error: {str(e)}")

def test_embedder_and_generator():
    print("\n--- Testing embedder.json configuration ---")
    embedder_config = load_embedder_config()
    print("Loaded embedder config:", embedder_config)
    # Test embedder (OpenAI as example)
    embedder = None
    if "embedder" in embedder_config and "model_client" in embedder_config["embedder"]:
        client_class = embedder_config["embedder"]["model_client"]
        kwargs = embedder_config["embedder"].get("initialize_kwargs", {})
        model_kwargs = embedder_config["embedder"].get("model_kwargs", {})
        embedder = client_class(**kwargs)
        print("Testing embedding...")
        # 使用convert_inputs_to_api_kwargs和call方法代替embed
        api_kwargs = embedder.convert_inputs_to_api_kwargs(
            input=["hello world", "deepwiki test"], 
            model_kwargs=model_kwargs,
            model_type=ModelType.EMBEDDER
        )
        result = embedder.call(api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER)
        print("Embedding result:", result)
    else:
        print("No valid embedder config found.")

    print("\n--- Testing generator.json configuration ---")
    generator_config = load_generator_config()
    print("Loaded generator config:", generator_config)
    # Test generator (OpenAI as example)
    if "providers" in generator_config and "openai" in generator_config["providers"]:
        provider = generator_config["providers"]["openai"]
        client_class = provider.get("model_client")
        if client_class:
            model = provider.get("default_model", "gpt-4o")
            model_kwargs = provider["models"].get(model, {})
            generator = client_class()
            print("Testing chat/generation...")
            messages = [
                {"role": "system", "content": "You are a coding assistant that talks like a pirate."},
                {"role": "user", "content": "How do I check if a Python object is an instance of a class?"}
            ]
            if hasattr(generator, "chat"):
                result = generator.chat(messages, {"model": model, **model_kwargs})
                print("Chat result:", result)
            else:
                print("Generator client does not support chat method.")
        else:
            print("No valid generator client found in config.")
    else:
        print("No valid openai provider in generator config.")
        
    # 测试Qwen客户端
    print("\n--- Testing Qwen API client ---")
    try:
        qwen_api_key = os.getenv("QWEN_API_KEY")
        if qwen_api_key:
            qwen_client = QwenClient()
            print("Testing Qwen chat...")
            
            # 测试聊天生成
            api_kwargs = qwen_client.convert_inputs_to_api_kwargs(
                input=[
                    {"role": "system", "content": "你是一个有用的AI助手。"},
                    {"role": "user", "content": "请介绍一下通义千问大模型"}
                ],
                model_kwargs={"model": "qwen-plus"},
                model_type=ModelType.LLM
            )
            result = qwen_client.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
            print("Qwen chat result:", result.choices[0].message.content)
            
            # 测试文本嵌入
            print("\nTesting Qwen embedding...")
            api_kwargs = qwen_client.convert_inputs_to_api_kwargs(
                input=["hello world", "deepwiki test"],
                model_kwargs={"model": "text-embedding-v3"},
                model_type=ModelType.EMBEDDER
            )
            result = qwen_client.call(api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER)
            print(f"Qwen embedding result: {len(result.data)} embeddings generated")
        else:
            print("QWEN_API_KEY not set, skipping Qwen API test")
    except Exception as e:
        print(f"Error testing Qwen API: {str(e)}")

if __name__ == "__main__":
    # 首先加载环境变量
    load_env()
    
    if len(sys.argv) == 1:
        test_embedder_and_generator()
    else:
        # fallback to original streaming test
        if len(sys.argv) < 3:
            print("Usage: python test_api.py <repo_url> <query> [file_path]")
            sys.exit(1)
        repo_url = sys.argv[1]
        query = sys.argv[2]
        file_path = sys.argv[3] if len(sys.argv) > 3 else None
        test_streaming_endpoint(repo_url, query, file_path)
