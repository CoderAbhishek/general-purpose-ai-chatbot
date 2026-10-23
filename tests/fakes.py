"""Test doubles shared by several test files."""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class StreamingFakeModel(BaseChatModel):
    """A real LangChain chat model with scripted replies.

    Unlike a plain fake, it streams text word by word through LangChain's callbacks,
    exactly the path stream_mode="messages" listens to. Tool-call replies are sent as
    tool-call chunks, as a real model would.
    """

    replies: list = Field(default_factory=list)
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "streaming-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _next_reply(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next_reply(messages).model_copy())])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self._next_reply(messages)
        if reply.tool_calls:
            chunks = [
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                for i, c in enumerate(reply.tool_calls)
            ]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
            return
        for i, word in enumerate(reply.content.split(" ")):
            text = word if i == 0 else " " + word
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk
