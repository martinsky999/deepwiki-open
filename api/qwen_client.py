"""Qwen ModelClient integration."""

import os
import json
import logging
import backoff
from typing import Dict, Sequence, Optional, Any, List, Generator, Union, TypeVar, Callable

import requests
from requests.exceptions import RequestException, Timeout

from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    EmbedderOutput,
    CompletionUsage,
    GeneratorOutput,
)
from adalflow.components.model_client.utils import parse_embedding_response

from openai import OpenAI, AsyncOpenAI, Stream
from openai.types import (
    Completion,
    CreateEmbeddingResponse,
    Image,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletion, ChatCompletionMessage
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)

log = logging.getLogger(__name__)
T = TypeVar("T")

# 辅助函数用于处理响应
def get_first_message_content(completion: ChatCompletion) -> str:
    """获取聊天完成响应中第一条消息的内容"""
    log.debug(f"raw completion: {completion}")
    return completion.choices[0].message.content

def parse_stream_response(completion: ChatCompletionChunk) -> str:
    """解析流式API响应"""
    return completion.choices[0].delta.content

def handle_streaming_response(generator: Stream[ChatCompletionChunk]):
    """处理流式响应"""
    for completion in generator:
        log.debug(f"Raw chunk completion: {completion}")
        parsed_content = parse_stream_response(completion)
        yield parsed_content

class QwenClient(ModelClient):
    __doc__ = r"""Qwen API客户端组件。

    使用OpenAI兼容模式与Qwen API交互，支持文本嵌入和聊天完成API。

    用户可以:
    1. 通过传递`QwenClient()`作为`model_client`简化`Embedder`和`Generator`组件的使用。
    2. 将此类作为参考，创建自己的API客户端或通过复制和修改代码扩展此类。

    参数:
        api_key (Optional[str], optional): Qwen API密钥。默认为`None`。
        chat_completion_parser (Callable[[Completion], Any], optional): 解析聊天完成的函数。默认为`None`。
        base_url (str): API基础URL。默认为`"https://dashscope.aliyuncs.com/compatible-mode/v1"`。
        env_api_key_name (str): API密钥的环境变量名。默认为`"QWEN_API_KEY"`。
        env_base_url_name (str): API基础URL的环境变量名。默认为`"QWEN_API_BASE_URL"`。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        base_url: Optional[str] = None,
        env_base_url_name: str = "QWEN_API_BASE_URL",
        env_api_key_name: str = "QWEN_API_KEY",
    ):
        """初始化Qwen客户端。

        建议设置QWEN_API_KEY环境变量而不是作为参数传递。

        参数:
            api_key (Optional[str], optional): Qwen API密钥。默认为None。
            base_url (str): API基础URL。
            env_api_key_name (str): API密钥的环境变量名。默认为`"QWEN_API_KEY"`。
        """
        super().__init__()
        self._api_key = api_key
        self._env_api_key_name = env_api_key_name
        self._env_base_url_name = env_base_url_name
        self.base_url = base_url or os.getenv(self._env_base_url_name, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.sync_client = self.init_sync_client()
        self.async_client = None  # 仅在调用异步方法时初始化
        self.chat_completion_parser = (
            chat_completion_parser or get_first_message_content
        )
        self._api_kwargs = {}  # 调用API时添加API参数

    def init_sync_client(self):
        """初始化同步客户端"""
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"环境变量 {self._env_api_key_name} 必须设置"
            )
        return OpenAI(api_key=api_key, base_url=self.base_url)

    def init_async_client(self):
        """初始化异步客户端"""
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"环境变量 {self._env_api_key_name} 必须设置"
            )
        return AsyncOpenAI(api_key=api_key, base_url=self.base_url)

    def parse_chat_completion(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> "GeneratorOutput":
        """解析聊天完成响应"""
        log.debug(f"completion: {completion}, parser: {self.chat_completion_parser}")
        try:
            data = self.chat_completion_parser(completion)
        except Exception as e:
            log.error(f"解析完成时出错: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=completion)

        try:
            usage = self.track_completion_usage(completion)
            return GeneratorOutput(
                data=None, error=None, raw_response=data, usage=usage
            )
        except Exception as e:
            log.error(f"跟踪完成使用情况时出错: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=data)

    def track_completion_usage(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> CompletionUsage:
        """跟踪完成使用情况"""
        try:
            usage: CompletionUsage = CompletionUsage(
                completion_tokens=completion.usage.completion_tokens,
                prompt_tokens=completion.usage.prompt_tokens,
                total_tokens=completion.usage.total_tokens,
            )
            return usage
        except Exception as e:
            log.error(f"跟踪完成使用情况时出错: {e}")
            return CompletionUsage(
                completion_tokens=None, prompt_tokens=None, total_tokens=None
            )

    def parse_embedding_response(
        self, response: CreateEmbeddingResponse
    ) -> EmbedderOutput:
        """解析嵌入响应"""
        try:
            return parse_embedding_response(response)
        except Exception as e:
            log.error(f"解析嵌入响应时出错: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)

    def convert_inputs_to_api_kwargs(
        self, input: Any = None, model_kwargs: Dict = None, model_type: ModelType = None
    ) -> Dict:
        """Convert inputs to API kwargs for Qwen API."""
        model_kwargs = model_kwargs or {}
        api_kwargs = {}
        
        if model_type == ModelType.LLM:
            # Handle LLM generation
            api_kwargs["model"] = model_kwargs.get("model", "qwen-plus")
            
            # Convert input to messages format if it's a string
            if isinstance(input, str):
                api_kwargs["messages"] = [{"role": "user", "content": input}]
            elif isinstance(input, list) and all(isinstance(msg, dict) for msg in input):
                api_kwargs["messages"] = input
            else:
                raise ValueError(f"Unsupported input format for Qwen: {type(input)}")
            
            # Add model parameters, but remove top_k which is not supported by Qwen API
            if "temperature" in model_kwargs:
                api_kwargs["temperature"] = model_kwargs["temperature"]
            if "top_p" in model_kwargs:
                api_kwargs["top_p"] = model_kwargs["top_p"]
            if "stream" in model_kwargs:
                api_kwargs["stream"] = model_kwargs["stream"]
            
            # Remove any unsupported parameters
            if "top_k" in api_kwargs:
                del api_kwargs["top_k"]
            
            return api_kwargs
        
        elif model_type == ModelType.EMBEDDER:
            # Handle embedding generation
            if isinstance(input, str):
                input = [input]
            
            if not isinstance(input, (list, tuple)):
                raise ValueError(f"Unsupported input format for embeddings: {type(input)}")
            
            api_kwargs["model"] = model_kwargs.get("model", "text-embedding-v1")
            api_kwargs["input"] = input
            
            # Add additional parameters if provided
            if "dimensions" in model_kwargs:
                api_kwargs["dimensions"] = model_kwargs["dimensions"]
            if "encoding_format" in model_kwargs:
                api_kwargs["encoding_format"] = model_kwargs["encoding_format"]
            
            return api_kwargs
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
            RequestException,
            Timeout,
        ),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """同步调用API"""
        log.info(f"api_kwargs: {api_kwargs}")
        self._api_kwargs = api_kwargs
        
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings.create(**api_kwargs)
        elif model_type == ModelType.LLM:
            if "stream" in api_kwargs and api_kwargs.get("stream", False):
                log.debug("流式调用")
                self.chat_completion_parser = handle_streaming_response
                return self.sync_client.chat.completions.create(**api_kwargs)
            else:
                log.debug("非流式调用")
                return self.sync_client.chat.completions.create(**api_kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
            RequestException,
            Timeout,
        ),
        max_time=5,
    )
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        """异步调用API"""
        self._api_kwargs = api_kwargs
        if self.async_client is None:
            self.async_client = self.init_async_client()
            
        if model_type == ModelType.EMBEDDER:
            return await self.async_client.embeddings.create(**api_kwargs)
        elif model_type == ModelType.LLM:
            return await self.async_client.chat.completions.create(**api_kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        """从字典创建对象"""
        obj = super().from_dict(data)
        # 重新创建客户端
        obj.sync_client = obj.init_sync_client()
        obj.async_client = None
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """将对象转换为字典"""
        # 排除不可序列化的对象
        exclude = [
            "sync_client",
            "async_client",
        ]
        output = super().to_dict(exclude=exclude)
        return output


# 示例用法
if __name__ == "__main__":
    from adalflow.core import Generator
    from adalflow.utils import setup_env

    setup_env()
    
    # 测试文本生成
    gen = Generator(
        model_client=QwenClient(),
        model_kwargs={"model": "qwen-plus", "stream": False},
    )
    gen_response = gen(prompt_kwargs={"input_str": "你好，请介绍一下通义千问大模型"})
    print(f"生成响应: {gen_response}") 