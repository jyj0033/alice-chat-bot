"""OpenAI Responses API Provider（/v1/responses）。

为什么需要它：MiniMax 的 Server Tools（服务端 `web_search`）只在
Anthropic Messages API 与 OpenAI Responses API 上可用，**不支持**
`/v1/chat/completions`。本 Provider 让正文回复走 Responses 端点，
模型可在单次请求内自行联网搜索并作答，无需外部搜索后端、无需判断器、
无需多轮回传 tool_use / tool_result。

与 Chat Completions 的关键差异（实现时须注意）：
  - 请求：`messages` → `input`；system 消息 → 顶层 `instructions`（合并成一条）
  - 工具：扁平结构 `{"type":"function","name":...}`，不是嵌套的 `function` 对象
  - 服务端 `web_search` **按需声明**：本 Provider 不自作主张往每个请求里塞 web_search，
    由上层在判断器认为需要查资料时，把 `{"type":"web_search"}` 加进 request.tools。
  - 响应：`output` 数组（reasoning / web_search_call / function_call / message），
          最终文本在顶层 `output_text`；message 片段里的 `annotations`
          （每条引用含来源标题/摘要，实测可达 50 条）会让该片段体积虚高，
          真正正文只在 `output_text`，因此解析优先走顶层字段。
  - **没有 `stop` 参数**（Chat Completions 用来注入协议边界的 `]<]minimax` 在此失效）
  - `max_output_tokens` **把 reasoning token 计入**，且 usage 里查不到 reasoning 明细，
    因此对预算要做下限保护，否则回复会被推理挤到截断。
  - `reasoning: {"effort": ...}` 是唯一能约束推理占用的手段（本 Provider 默认传 low）：
    不传时模型自适应，实测把整 2000 预算烧在 reasoning 上、status=incomplete 且
    一条 message 都不产出；传 "low" 后同等任务 output 降到 188 并正常产出。
    `"none"` 会被端点拒绝——该模型要求 adaptive thinking。
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Optional

import aiohttp

from .base import LLMProvider, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

# reasoning token 会被计入 max_output_tokens 且无法从 usage 观测。
# 实测两种失败样本：
#   - 生产值 500 时，一次联网回复用掉 493（98.6%），差一点就截断；
#   - 1200 时仍出现过一次「整 1200 全被 reasoning 吃掉、status=incomplete、
#     一条 message 都没产出」——等于这次回复彻底丢失。
# 上限制不消耗额度（按实际用量计费），所以宁可放宽给推理留余量。
MIN_OUTPUT_TOKENS = 2000

# 服务端工具类型：由上层按需塞进 request.tools，此处只按能力开关放行。
_SERVER_TOOL_TYPES = {"web_search"}


class ResponsesProvider(LLMProvider):
    """OpenAI Responses API Provider（支持服务端 web_search）。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        base_url = (config.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        self.model = config.get("model", "gpt-4o")
        self.timeout = float(config.get("timeout", 120))
        # 服务端联网搜索「能力开关」：决定本 Provider 是否具备服务端搜索。
        # 注意它**不等于**每次请求都声明 web_search —— 是否声明由上层按需决定
        # （generator 只在检索判断器认为需要查资料时，把 {"type":"web_search"}
        # 塞进 request.tools）。常驻声明会让「晚饭吃啥」这类闲聊也触发多轮联网，
        # 实测输入 token 从 233 膨胀到 2 万~68 万。
        self.web_search = bool(config.get("web_search", False))
        configured = int(config.get("max_output_tokens") or config.get("max_tokens") or 0)
        self.max_output_tokens = max(MIN_OUTPUT_TOKENS, configured)
        # reasoning 档位，默认 low：不传时模型自行决定推理长度，辅助任务会把
        # 整个输出预算烧在 reasoning 上、一条 message 都不产出（线上约 71% 失败）。
        # 显式配置成 null 或空串可关闭，恢复模型自适应（详见 _build_body）。
        configured_effort = config.get("reasoning_effort", "low")
        self.reasoning_effort = (
            str(configured_effort).strip().lower() if configured_effort else None
        )
        self._warned_stop = False
        self.url = f"{base_url}/responses"
        logger.info(
            "Responses 提供商已初始化：地址=%s，模型=%s，服务端搜索=%s，"
            "输出上限=%d，推理档位=%s",
            self.url, self.model, "开" if self.web_search else "关",
            self.max_output_tokens, self.reasoning_effort or "自适应",
        )

    @property
    def provider_name(self) -> str:
        return "responses"

    @property
    def supports_server_search(self) -> bool:
        """本 Provider 是否具备服务端联网搜索能力（供上层决定「按需声明」）。

        上层据此做两件事：① 关掉「外部搜索 + 资料注入」那一路，避免同一句话
        被搜两次；② 在检索判断器认为需要查资料时，把 {"type":"web_search"}
        加进本次请求的 tools。
        """
        return self.web_search

    # ------------------------------------------------------------------ 请求构建

    @staticmethod
    def _split_system(messages) -> tuple[str, list]:
        """system 消息合并成 instructions，其余转成 input 数组。"""
        instructions: list[str] = []
        items: list[dict] = []
        for msg in messages or []:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")
            if role == "system":
                text = content
                if isinstance(content, list):
                    text = "\n".join(
                        str(p.get("text", "")) for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                text = str(text or "").strip()
                if text:
                    instructions.append(text)
                continue
            if role == "tool":
                # Responses 用 function_call_output 回传工具结果。
                items.append({
                    "type": "function_call_output",
                    "call_id": getattr(msg, "tool_call_id", "") or "",
                    "output": content if isinstance(content, str) else json.dumps(
                        content, ensure_ascii=False),
                })
                continue
            items.append(ResponsesProvider._to_input_item(msg))
        return "\n\n".join(instructions), items

    @staticmethod
    def _to_input_item(msg) -> dict:
        """单条消息 → Responses input item（处理多模态片段）。"""
        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", "")
        item: dict = {"role": role}
        if isinstance(content, str) or content is None:
            item["content"] = content or ""
            return item
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                parts.append({"type": "input_text", "text": part.get("text", "")})
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url:
                    parts.append({"type": "input_image", "image_url": url})
            elif ptype in ("input_text", "input_image"):
                parts.append(part)
        item["content"] = parts or ""
        return item

    def _build_tools(self, tools) -> list[dict]:
        """OpenAI Chat 工具定义 → Responses 扁平结构；服务端工具按能力放行。

        服务端 web_search **不在这里主动声明**——由上层按需塞进 request.tools
        （判断器认为这条消息需要查资料时才加）。这里只做闸门：能力打开时放行，
        关闭时剔除，防止配置与调用不一致时误开联网。
        """
        out: list[dict] = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            ttype = str(tool.get("type") or "")
            # 服务端工具（web_search）：按能力开关放行，不主动追加
            if ttype in _SERVER_TOOL_TYPES:
                if self.web_search:
                    out.append(tool)
                else:
                    logger.warning(
                        "[Responses] 收到服务端工具 %s 但 web_search 能力未开，已忽略", ttype
                    )
                continue
            # 其它非 function 类型（历史遗留形态）→ 直通
            if ttype and ttype != "function":
                out.append(tool)
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else None
            if fn:
                item = {
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            else:
                # 已是扁平形态
                item = {k: v for k, v in tool.items() if k != "type"}
                item["type"] = "function"
            if item.get("name"):
                out.append(item)
        return out

    def _build_body(self, request: ChatRequest) -> dict:
        """组装 Responses 请求体（独立出来便于离线断言字段）。

        `reasoning.effort` 是这里的保命参数：不传时模型会自行决定推理长度，
        实测「从 64 条词表里挑出不合语境的条目」这种辅助任务会把
        max_output_tokens=2000 全部烧在 reasoning 上，status=incomplete、
        一条 message 都不产出（线上辅助任务约 71% 失败）。传 effort=low 后
        同等任务 output 降到 188、正常产出。effort=none 会被端点拒绝
        （该模型要求 adaptive thinking）。
        """
        instructions, items = self._split_system(request.messages)
        body: dict = {
            "model": request.model or self.model,
            "input": items,
            "max_output_tokens": max(
                MIN_OUTPUT_TOKENS,
                int(getattr(request, "max_tokens", 0) or 0),
                self.max_output_tokens,
            ),
        }
        if instructions:
            body["instructions"] = instructions
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        tools = self._build_tools(request.tools)
        if tools:
            body["tools"] = tools
            # 观测用：明确标出本轮是否声明了服务端搜索，方便对照 prompt_tokens 分布。
            if any(
                isinstance(t, dict) and t.get("type") in _SERVER_TOOL_TYPES for t in tools
            ):
                logger.info("[Responses] 本轮按需声明服务端 web_search")
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """发送请求（单次请求内完成服务端搜索 + 作答）。"""
        body = self._build_body(request)
        # Responses API 没有 stop 参数。Chat Completions 那套协议边界停止序列
        # （generator._PROTOCOL_STOP_SEQUENCES）在此不可用，只能靠文本清洗兜底。
        if request.stop and not self._warned_stop:
            self._warned_stop = True
            logger.warning(
                "[Responses] 端点不支持 stop 序列，已忽略 %s；协议边界泄漏改由文本清洗处理",
                request.stop,
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url, headers=headers, json=body,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    raw = await resp.text()
                    if resp.status != 200:
                        logger.error(
                            "[Responses] HTTP %s - %s", resp.status, raw[:500]
                        )
                        raise RuntimeError(
                            f"Responses API error: {resp.status} - {raw[:500]}"
                        )
            payload = json.loads(raw)
        except aiohttp.ClientError as exc:
            logger.error("[Responses] 客户端请求出错：%s", exc)
            raise

        return self._parse(payload)

    def _parse(self, payload: dict) -> ChatResponse:
        """解析 output 数组：取 output_text、抽 function_call、忽略 reasoning。"""
        items = [i for i in (payload.get("output") or []) if isinstance(i, dict)]

        # 最终文本：优先顶层 output_text，回退拼 message 里的 output_text
        content = payload.get("output_text") or ""
        if not content:
            chunks = []
            for item in items:
                if item.get("type") != "message":
                    continue
                for part in (item.get("content") or []):
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        chunks.append(part.get("text", ""))
            content = "".join(chunks)

        # 客户端工具调用（send_meme 等）→ 与 OpenAI/Anthropic 一致的内部结构
        tool_calls = []
        for item in items:
            if item.get("type") != "function_call":
                continue
            raw_args = item.get("arguments")
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args or "{}")
                except json.JSONDecodeError:
                    arguments = {"raw": raw_args}
            else:
                arguments = raw_args if isinstance(raw_args, dict) else {}
            tool_calls.append({
                "id": item.get("call_id") or item.get("id") or "",
                "name": item.get("name") or "",
                "arguments": arguments,
            })

        status = payload.get("status") or "completed"
        incomplete = payload.get("incomplete_details") or {}
        if status == "incomplete":
            logger.warning(
                "[Responses] 回复未完成：%s（输出上限 %d，可能被 reasoning 挤占）",
                incomplete or "未知原因", self.max_output_tokens,
            )
        err = payload.get("error")
        if err:
            logger.error("[Responses] 返回错误：%s", json.dumps(err, ensure_ascii=False)[:300])

        usage = payload.get("usage") or {}
        types = [i.get("type") for i in items]
        logger.info(
            "[Responses] 停止=%s，输出项=%s，文本长度=%d，工具调用=%d（%s）",
            status, types, len(content), len(tool_calls),
            [tc.get("name") for tc in tool_calls],
        )

        return ChatResponse(
            content=content,
            model=payload.get("model") or self.model,
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            finish_reason=status,
            raw_response=payload,
            tool_calls=tool_calls,
        )

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        """未实现：全项目无调用点（流式回复未启用）。"""
        raise NotImplementedError("ResponsesProvider 暂不支持流式输出")
        yield ""  # pragma: no cover 保持异步生成器语义

    async def close(self) -> None:
        pass


def create_responses_provider(config: dict) -> ResponsesProvider:
    return ResponsesProvider(config)
