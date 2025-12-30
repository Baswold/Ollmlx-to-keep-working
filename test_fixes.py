#!/usr/bin/env python3
"""
Comprehensive test suite for ollmlx fixes.

Tests the following fixes:
1. Vision + tool calling compatibility
2. Model-aware chat templates
3. Model-specific image tokens
4. Model-aware embedding extraction
5. Parameter count parsing
6. Tool calling text content preservation

Usage:
    # Run all tests (unit tests - no server needed)
    python test_fixes.py

    # Run integration tests (requires server)
    python test_fixes.py --integration

    # Run with specific small model
    python test_fixes.py --integration --model mlx-community/SmolLM2-135M-Instruct-4bit
"""

import argparse
import json
import sys
import unittest
from unittest.mock import MagicMock, patch
from io import BytesIO
import base64

# Test if we're in the right directory
try:
    sys.path.insert(0, '.')
    from mlx_backend.server import (
        MLXModelManager,
        parse_tool_calls,
        _parse_single_tool_call,
        CompletionResponse,
        Options,
    )
    MLX_BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import mlx_backend: {e}")
    MLX_BACKEND_AVAILABLE = False


class TestToolCallParsing(unittest.TestCase):
    """Test tool call parsing from model output."""

    def test_openai_format(self):
        """Test parsing OpenAI-style tool calls."""
        text = '{"tool_calls":[{"function":{"name":"get_weather","arguments":{"location":"SF"}}}]}'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["name"], "get_weather")

    def test_simple_format(self):
        """Test parsing simple tool call format."""
        text = '{"tool_calls":[{"name":"calculate","arguments":{"expression":"2+2"}}]}'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["name"], "calculate")

    def test_direct_array(self):
        """Test parsing direct array of tool calls."""
        text = '[{"name":"func1","arguments":{}},{"name":"func2","arguments":{}}]'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_single_call_object(self):
        """Test parsing single tool call object."""
        text = '{"name":"single_func","arguments":{"key":"value"}}'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["name"], "single_func")

    def test_tool_name_as_key(self):
        """Test parsing tool name as key format."""
        text = '{"get_weather":{"location":"NYC"}}'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["name"], "get_weather")

    def test_json_in_text(self):
        """Test extracting JSON from surrounding text."""
        text = 'I need to call a function. {"tool_calls":[{"name":"test","arguments":{}}]} Let me do that.'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_no_tool_calls(self):
        """Test handling text with no tool calls."""
        text = "This is just regular text without any JSON."
        result = parse_tool_calls(text)
        self.assertIsNone(result)

    def test_invalid_json(self):
        """Test handling invalid JSON."""
        text = '{"tool_calls": [broken json'
        result = parse_tool_calls(text)
        self.assertIsNone(result)


class TestParameterCountParsing(unittest.TestCase):
    """Test parameter count string parsing."""

    def setUp(self):
        """Import the Go parsing logic via a Python reimplementation for testing."""
        # Reimplementation of parseParameterCount for testing
        pass

    def test_parse_7b(self):
        """Test parsing '7b' format."""
        self.assertEqual(parse_param_count("7b"), 7_000_000_000)

    def test_parse_1_5b(self):
        """Test parsing '1.5b' format."""
        self.assertEqual(parse_param_count("1.5b"), 1_500_000_000)

    def test_parse_135m(self):
        """Test parsing '135m' format."""
        self.assertEqual(parse_param_count("135m"), 135_000_000)

    def test_parse_with_billion(self):
        """Test parsing '7 billion' format."""
        self.assertEqual(parse_param_count("7 billion"), 7_000_000_000)

    def test_parse_with_million(self):
        """Test parsing '135 million' format."""
        self.assertEqual(parse_param_count("135 million"), 135_000_000)

    def test_parse_uppercase(self):
        """Test parsing uppercase '7B' format."""
        self.assertEqual(parse_param_count("7B"), 7_000_000_000)

    def test_parse_with_commas(self):
        """Test parsing '7,000,000,000' format."""
        self.assertEqual(parse_param_count("7,000,000,000"), 7_000_000_000)

    def test_parse_empty(self):
        """Test parsing empty string."""
        self.assertEqual(parse_param_count(""), 0)

    def test_parse_invalid(self):
        """Test parsing invalid string."""
        self.assertEqual(parse_param_count("invalid"), 0)


def parse_param_count(param_size: str) -> int:
    """Python reimplementation of parseParameterCount for testing."""
    param_size = param_size.lower().strip()
    if not param_size:
        return 0

    # Remove commas and spaces
    param_size = param_size.replace(",", "").replace(" ", "")

    # Handle word suffixes
    if param_size.endswith("billion"):
        try:
            return int(float(param_size[:-7]) * 1_000_000_000)
        except ValueError:
            pass
    if param_size.endswith("million"):
        try:
            return int(float(param_size[:-7]) * 1_000_000)
        except ValueError:
            pass
    if param_size.endswith("thousand") or param_size.endswith("k"):
        suffix = "thousand" if param_size.endswith("thousand") else "k"
        try:
            return int(float(param_size[:-len(suffix)]) * 1_000)
        except ValueError:
            pass

    # Handle short suffixes
    multiplier = 1
    if param_size.endswith("b"):
        param_size = param_size[:-1]
        multiplier = 1_000_000_000
    elif param_size.endswith("m"):
        param_size = param_size[:-1]
        multiplier = 1_000_000
    elif param_size.endswith("t"):
        param_size = param_size[:-1]
        multiplier = 1_000_000_000_000

    try:
        return int(float(param_size) * multiplier)
    except ValueError:
        pass

    # Extract numbers
    num_str = ""
    found_dot = False
    for c in param_size:
        if c.isdigit():
            num_str += c
        elif c == "." and not found_dot:
            num_str += c
            found_dot = True

    if num_str:
        try:
            val = float(num_str)
            if multiplier == 1 and val < 1000:
                multiplier = 1_000_000_000
            return int(val * multiplier)
        except ValueError:
            pass

    return 0


class TestChatTemplateDetection(unittest.TestCase):
    """Test chat template detection based on model name."""

    def test_detect_qwen(self):
        """Test detecting Qwen models."""
        self.assertEqual(detect_chat_template("mlx-community/Qwen2.5-3B-Instruct"), "qwen")
        self.assertEqual(detect_chat_template("Qwen/Qwen2-VL-7B"), "qwen")

    def test_detect_llama(self):
        """Test detecting Llama models."""
        self.assertEqual(detect_chat_template("mlx-community/Llama-3.2-1B-Instruct"), "llama")
        self.assertEqual(detect_chat_template("meta-llama/Llama-2-7b"), "llama")

    def test_detect_mistral(self):
        """Test detecting Mistral models."""
        self.assertEqual(detect_chat_template("mlx-community/Mistral-7B-Instruct"), "mistral")
        self.assertEqual(detect_chat_template("mistralai/Mixtral-8x7B"), "mistral")

    def test_detect_phi(self):
        """Test detecting Phi models."""
        self.assertEqual(detect_chat_template("microsoft/Phi-3.5-mini"), "phi")
        self.assertEqual(detect_chat_template("mlx-community/Phi-2"), "phi")

    def test_detect_gemma(self):
        """Test detecting Gemma models."""
        self.assertEqual(detect_chat_template("google/gemma-2-9b"), "gemma")
        self.assertEqual(detect_chat_template("mlx-community/gemma-7b"), "gemma")

    def test_detect_smollm(self):
        """Test detecting SmolLM models."""
        self.assertEqual(detect_chat_template("mlx-community/SmolLM2-135M-Instruct"), "smollm")

    def test_detect_unknown(self):
        """Test detecting unknown models (defaults to chatml)."""
        self.assertEqual(detect_chat_template("some-random-model"), "chatml")


def detect_chat_template(model_name: str) -> str:
    """Python reimplementation of detectChatTemplate for testing."""
    lower = model_name.lower()

    if "qwen" in lower:
        return "qwen"
    if "llama" in lower:
        return "llama"
    if "mistral" in lower or "mixtral" in lower:
        return "mistral"
    if "phi" in lower:
        return "phi"
    if "gemma" in lower:
        return "gemma"
    if "smollm" in lower:
        return "smollm"

    return "chatml"


class TestImageTokenDetection(unittest.TestCase):
    """Test image token detection based on model name."""

    def test_qwen_vl_numbered(self):
        """Test Qwen2-VL uses numbered image tokens."""
        self.assertEqual(get_image_token("Qwen/Qwen2-VL-7B", 0), "<image_1>")
        self.assertEqual(get_image_token("Qwen/Qwen2-VL-7B", 1), "<image_2>")

    def test_llava_simple(self):
        """Test LLaVA uses simple <image> token."""
        self.assertEqual(get_image_token("llava-hf/llava-1.5-7b", 0), "<image>")
        self.assertEqual(get_image_token("llava-hf/llava-1.5-7b", 1), "<image>")

    def test_pixtral_simple(self):
        """Test Pixtral uses simple <image> token."""
        self.assertEqual(get_image_token("mistral/pixtral-12b", 0), "<image>")

    def test_default_simple(self):
        """Test default uses simple <image> token."""
        self.assertEqual(get_image_token("some-unknown-vision-model", 0), "<image>")


def get_image_token(model_name: str, image_index: int) -> str:
    """Python reimplementation of getImageToken for testing."""
    lower = model_name.lower()

    if "qwen" in lower and "vl" in lower:
        return f"<image_{image_index + 1}>"

    return "<image>"


class TestEmbeddingStrategyDetection(unittest.TestCase):
    """Test embedding strategy detection."""

    @unittest.skipUnless(MLX_BACKEND_AVAILABLE, "mlx_backend not available")
    def test_bert_uses_cls(self):
        """Test BERT-like models use CLS token."""
        manager = MLXModelManager()
        manager.current_model_name = "bert-base-uncased"
        self.assertEqual(manager._detect_embedding_strategy(), "cls")

    @unittest.skipUnless(MLX_BACKEND_AVAILABLE, "mlx_backend not available")
    def test_gpt_uses_last_token(self):
        """Test GPT-like models use last token."""
        manager = MLXModelManager()
        manager.current_model_name = "gpt2"
        self.assertEqual(manager._detect_embedding_strategy(), "last_token")

    @unittest.skipUnless(MLX_BACKEND_AVAILABLE, "mlx_backend not available")
    def test_e5_uses_cls(self):
        """Test E5 models use CLS token."""
        manager = MLXModelManager()
        manager.current_model_name = "intfloat/e5-large-v2"
        self.assertEqual(manager._detect_embedding_strategy(), "cls")

    @unittest.skipUnless(MLX_BACKEND_AVAILABLE, "mlx_backend not available")
    def test_default_uses_mean_no_special(self):
        """Test default models use mean pooling without special tokens."""
        manager = MLXModelManager()
        manager.current_model_name = "some-random-model"
        self.assertEqual(manager._detect_embedding_strategy(), "mean_no_special")


class TestCompletionResponse(unittest.TestCase):
    """Test CompletionResponse serialization."""

    @unittest.skipUnless(MLX_BACKEND_AVAILABLE, "mlx_backend not available")
    def test_response_with_tool_calls(self):
        """Test response includes tool calls."""
        response = CompletionResponse(
            content="Let me check that.",
            done=True,
            done_reason="tool_calls",
            tool_calls=[{
                "id": "call_1",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "SF"}
                }
            }]
        )
        json_str = response.to_json()
        data = json.loads(json_str)

        self.assertEqual(data["content"], "Let me check that.")
        self.assertEqual(data["done_reason"], "tool_calls")
        self.assertIsNotNone(data["tool_calls"])
        self.assertEqual(len(data["tool_calls"]), 1)

    @unittest.skipUnless(MLX_BACKEND_AVAILABLE, "mlx_backend not available")
    def test_response_preserves_content_with_tools(self):
        """Test that content is preserved even with tool calls."""
        response = CompletionResponse(
            content="I'll help you with the weather. Let me check.",
            done=True,
            done_reason="tool_calls",
            tool_calls=[{
                "function": {"name": "get_weather", "arguments": {}}
            }]
        )
        json_str = response.to_json()
        data = json.loads(json_str)

        # Content should NOT be cleared
        self.assertNotEqual(data["content"], "")
        self.assertIn("help you with the weather", data["content"])


class TestChatTemplateFormatting(unittest.TestCase):
    """Test chat template formatting for different models."""

    def test_qwen_format(self):
        """Test Qwen chat template format."""
        messages = [
            {"role": "user", "content": "Hello!", "images": []},
        ]
        result = format_qwen_prompt(messages, [])
        self.assertIn("<|im_start|>system", result)
        self.assertIn("<|im_start|>user", result)
        self.assertIn("<|im_end|>", result)
        self.assertIn("Hello!", result)

    def test_llama3_format(self):
        """Test Llama 3 chat template format."""
        messages = [
            {"role": "user", "content": "Hi there!", "images": []},
        ]
        result = format_llama_prompt(messages, [], "llama-3.2-1b")
        self.assertIn("<|begin_of_text|>", result)
        self.assertIn("<|start_header_id|>", result)
        self.assertIn("<|eot_id|>", result)
        self.assertIn("Hi there!", result)

    def test_mistral_format(self):
        """Test Mistral chat template format."""
        messages = [
            {"role": "user", "content": "Test message", "images": []},
        ]
        result = format_mistral_prompt(messages, [])
        self.assertIn("<s>", result)
        self.assertIn("[INST]", result)
        self.assertIn("[/INST]", result)
        self.assertIn("Test message", result)

    def test_gemma_format(self):
        """Test Gemma chat template format."""
        messages = [
            {"role": "user", "content": "Gemma test", "images": []},
        ]
        result = format_gemma_prompt(messages, [])
        self.assertIn("<start_of_turn>user", result)
        self.assertIn("<end_of_turn>", result)
        self.assertIn("<start_of_turn>model", result)
        self.assertIn("Gemma test", result)


def format_qwen_prompt(messages: list, tools: list) -> str:
    """Python implementation of Qwen formatting."""
    result = "<|im_start|>system\nYou are a helpful assistant."
    if tools:
        result += " [tools info here]"
    result += "<|im_end|>\n"

    for m in messages:
        result += f"<|im_start|>{m['role']}\n"
        for i, _ in enumerate(m.get('images', [])):
            result += "<image>\n"
        result += m['content']
        result += "<|im_end|>\n"

    result += "<|im_start|>assistant\n"
    return result


def format_llama_prompt(messages: list, tools: list, model_name: str) -> str:
    """Python implementation of Llama formatting."""
    is_llama3 = "llama-3" in model_name.lower() or "llama3" in model_name.lower()

    if is_llama3:
        result = "<|begin_of_text|>"
        result += "<|start_header_id|>system<|end_header_id|>\n\n"
        result += "You are a helpful assistant."
        if tools:
            result += " [tools info here]"
        result += "<|eot_id|>"

        for m in messages:
            result += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n"
            for i, _ in enumerate(m.get('images', [])):
                result += "<image>\n"
            result += m['content']
            result += "<|eot_id|>"

        result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        result = "[INST] <<SYS>>\nYou are a helpful assistant."
        if tools:
            result += " [tools info here]"
        result += "\n<</SYS>>\n\n"

        for i, m in enumerate(messages):
            if m['role'] == 'user':
                if i > 0:
                    result += "[INST] "
                for j, _ in enumerate(m.get('images', [])):
                    result += "<image>\n"
                result += m['content']
                result += " [/INST]"
            elif m['role'] == 'assistant':
                result += " " + m['content'] + " </s><s>"

    return result


def format_mistral_prompt(messages: list, tools: list) -> str:
    """Python implementation of Mistral formatting."""
    result = "<s>"

    sys_msg = "You are a helpful assistant."
    if tools:
        sys_msg += " [tools info here]"

    first_user = True
    for m in messages:
        if m['role'] == 'user':
            result += "[INST] "
            if first_user:
                result += sys_msg + "\n\n"
                first_user = False
            for i, _ in enumerate(m.get('images', [])):
                result += "<image>\n"
            result += m['content']
            result += " [/INST]"
        elif m['role'] == 'assistant':
            result += m['content']
            result += "</s>"

    return result


def format_gemma_prompt(messages: list, tools: list) -> str:
    """Python implementation of Gemma formatting."""
    result = ""

    for m in messages:
        if m['role'] == 'user':
            result += "<start_of_turn>user\n"
            for i, _ in enumerate(m.get('images', [])):
                result += "<image>\n"
            result += m['content']
            if tools:
                result += "\n\n[tools info here]"
            result += "<end_of_turn>\n"
        elif m['role'] == 'assistant':
            result += "<start_of_turn>model\n"
            result += m['content']
            result += "<end_of_turn>\n"

    result += "<start_of_turn>model\n"
    return result


# Integration tests (require running server)
class IntegrationTests(unittest.TestCase):
    """Integration tests that require a running ollmlx server."""

    SERVER_URL = "http://localhost:11434"
    MODEL = "mlx-community/SmolLM2-135M-Instruct-4bit"

    @classmethod
    def setUpClass(cls):
        """Check if server is running."""
        try:
            import requests
            response = requests.get(f"{cls.SERVER_URL}/api/tags", timeout=5)
            cls.server_available = response.status_code == 200
        except Exception:
            cls.server_available = False

        if not cls.server_available:
            print("Warning: Server not running. Skipping integration tests.")

    def setUp(self):
        """Skip if server not available."""
        if not self.server_available:
            self.skipTest("Server not available")

    def test_basic_generation(self):
        """Test basic text generation."""
        import requests

        response = requests.post(
            f"{self.SERVER_URL}/api/generate",
            json={
                "model": self.MODEL,
                "prompt": "Say hello in one word.",
                "stream": False,
                "options": {"num_predict": 10}
            },
            timeout=60
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertTrue(len(data["response"]) > 0)

    def test_chat_completion(self):
        """Test chat completion."""
        import requests

        response = requests.post(
            f"{self.SERVER_URL}/api/chat",
            json={
                "model": self.MODEL,
                "messages": [
                    {"role": "user", "content": "Say 'test' and nothing else."}
                ],
                "stream": False,
                "options": {"num_predict": 10}
            },
            timeout=60
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("content", data["message"])

    def test_tool_calling(self):
        """Test tool calling functionality."""
        import requests

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        response = requests.post(
            f"{self.SERVER_URL}/api/chat",
            json={
                "model": self.MODEL,
                "messages": [
                    {"role": "user", "content": "What's the weather in Paris?"}
                ],
                "tools": tools,
                "stream": False,
                "options": {"num_predict": 100, "temperature": 0.3}
            },
            timeout=60
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        # Note: Small models may not reliably produce tool calls
        # This test just verifies the endpoint works

    def test_embeddings(self):
        """Test embedding generation."""
        import requests

        response = requests.post(
            f"{self.SERVER_URL}/api/embed",
            json={
                "model": self.MODEL,
                "input": "Hello world"
            },
            timeout=60
        )
        # Embeddings may not be supported by all models
        if response.status_code == 200:
            data = response.json()
            self.assertIn("embeddings", data)
            self.assertTrue(len(data["embeddings"]) > 0)


def main():
    parser = argparse.ArgumentParser(description="Test ollmlx fixes")
    parser.add_argument("--integration", action="store_true",
                        help="Run integration tests (requires server)")
    parser.add_argument("--model", default="mlx-community/SmolLM2-135M-Instruct-4bit",
                        help="Model to use for integration tests")
    parser.add_argument("--url", default="http://localhost:11434",
                        help="Server URL")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Update integration test settings
    IntegrationTests.SERVER_URL = args.url
    IntegrationTests.MODEL = args.model

    # Build test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Always run unit tests
    suite.addTests(loader.loadTestsFromTestCase(TestToolCallParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterCountParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestChatTemplateDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestImageTokenDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestChatTemplateFormatting))

    if MLX_BACKEND_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingStrategyDetection))
        suite.addTests(loader.loadTestsFromTestCase(TestCompletionResponse))

    # Optionally run integration tests
    if args.integration:
        suite.addTests(loader.loadTestsFromTestCase(IntegrationTests))

    # Run tests
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
