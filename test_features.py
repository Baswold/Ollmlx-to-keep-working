#!/usr/bin/env python3
"""
Test script for ollmlx image support and tool calling features.

Usage:
    # Start ollmlx server first:
    # ./ollmlx serve

    # Then run this test script:
    python test_features.py [--test-images] [--test-tools] [--all]

Requirements:
    pip install requests pillow
"""

import argparse
import base64
import json
import sys
from io import BytesIO

import requests

# Default server URL
OLLMLX_URL = "http://localhost:11434"


def create_test_image():
    """Create a simple test image (red square) as base64."""
    try:
        from PIL import Image
    except ImportError:
        print("PIL not installed. Install with: pip install pillow")
        return None

    # Create a simple 100x100 red image
    img = Image.new("RGB", (100, 100), color="red")

    # Add some text/pattern to make it more interesting
    for x in range(100):
        for y in range(100):
            if (x + y) % 20 < 10:
                img.putpixel((x, y), (255, 255, 0))  # Yellow diagonal stripes

    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def test_image_support_generate(model_name: str = "mlx-community/llava-1.5-7b-4bit"):
    """Test image support with the /api/generate endpoint."""
    print("\n" + "=" * 60)
    print("Testing Image Support with /api/generate")
    print("=" * 60)

    # Create test image
    image_b64 = create_test_image()
    if not image_b64:
        print("SKIP: Could not create test image")
        return False

    print(f"Model: {model_name}")
    print("Created test image: 100x100 with yellow diagonal stripes on red background")

    # Build request
    request = {
        "model": model_name,
        "prompt": "What colors do you see in this image? Describe the pattern.",
        "images": [image_b64],
        "stream": False,
        "options": {
            "num_predict": 100,
            "temperature": 0.7
        }
    }

    try:
        print("\nSending request to /api/generate...")
        response = requests.post(
            f"{OLLMLX_URL}/api/generate",
            json=request,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\nResponse: {result.get('response', 'No response')}")
            print("\nImage test: PASSED")
            return True
        else:
            print(f"\nError {response.status_code}: {response.text}")
            print("\nImage test: FAILED")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\nCould not connect to {OLLMLX_URL}")
        print("Make sure ollmlx server is running: ./ollmlx serve")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False


def test_image_support_chat(model_name: str = "mlx-community/llava-1.5-7b-4bit"):
    """Test image support with the /api/chat endpoint."""
    print("\n" + "=" * 60)
    print("Testing Image Support with /api/chat")
    print("=" * 60)

    # Create test image
    image_b64 = create_test_image()
    if not image_b64:
        print("SKIP: Could not create test image")
        return False

    print(f"Model: {model_name}")

    # Build request
    request = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "What do you see in this image?",
                "images": [image_b64]
            }
        ],
        "stream": False,
        "options": {
            "num_predict": 100,
            "temperature": 0.7
        }
    }

    try:
        print("\nSending request to /api/chat...")
        response = requests.post(
            f"{OLLMLX_URL}/api/chat",
            json=request,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            message = result.get("message", {})
            print(f"\nResponse: {message.get('content', 'No content')}")
            print("\nChat image test: PASSED")
            return True
        else:
            print(f"\nError {response.status_code}: {response.text}")
            print("\nChat image test: FAILED")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\nCould not connect to {OLLMLX_URL}")
        print("Make sure ollmlx server is running: ./ollmlx serve")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False


def test_tool_calling(model_name: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"):
    """Test tool calling functionality."""
    print("\n" + "=" * 60)
    print("Testing Tool Calling")
    print("=" * 60)

    print(f"Model: {model_name}")

    # Define test tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    # Build request
    request = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in San Francisco?"
            }
        ],
        "tools": tools,
        "stream": False,
        "options": {
            "num_predict": 200,
            "temperature": 0.3  # Lower temperature for more consistent tool calls
        }
    }

    try:
        print("\nSending request to /api/chat with tools...")
        response = requests.post(
            f"{OLLMLX_URL}/api/chat",
            json=request,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            message = result.get("message", {})
            tool_calls = message.get("tool_calls", [])
            content = message.get("content", "")

            print(f"\nContent: {content[:200]}..." if len(content) > 200 else f"\nContent: {content}")

            if tool_calls:
                print(f"\nTool calls detected: {len(tool_calls)}")
                valid_calls = 0
                for i, tc in enumerate(tool_calls):
                    func = tc.get("function", {})
                    name = func.get('name', '')
                    args = func.get('arguments', {})
                    print(f"  Tool {i+1}: {name}")
                    print(f"  Arguments: {args}")
                    # Verify the tool call is valid
                    if name == "get_weather" and isinstance(args, dict):
                        valid_calls += 1

                if valid_calls > 0:
                    print("\nTool calling test: PASSED")
                    return True
                else:
                    print("\nTool calls found but none are valid get_weather calls")
                    print("\nTool calling test: FAILED")
                    return False
            else:
                # Try to parse tool calls from content ourselves
                try:
                    # Look for JSON in the content
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        if "get_weather" in str(parsed) or "tool_calls" in parsed:
                            print("\nModel generated tool call JSON in content (server parsing issue)")
                            print(f"Parsed: {parsed}")
                            print("\nTool calling test: PARTIAL (server should parse this)")
                            return False  # This is not a pass - server should parse it
                except:
                    pass

                print("\nNo tool calls in response")
                print("\nTool calling test: FAILED (model may not support tool calling)")
                return False
        else:
            print(f"\nError {response.status_code}: {response.text}")
            print("\nTool calling test: FAILED")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\nCould not connect to {OLLMLX_URL}")
        print("Make sure ollmlx server is running: ./ollmlx serve")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False


def test_tool_calling_generate(model_name: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"):
    """Test tool calling with /api/generate endpoint."""
    print("\n" + "=" * 60)
    print("Testing Tool Calling with /api/generate")
    print("=" * 60)

    print(f"Model: {model_name}")

    # Define test tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    # Build request
    request = {
        "model": model_name,
        "prompt": "What is 25 * 17? Use the calculate tool to compute this.",
        "tools": tools,
        "stream": False,
        "options": {
            "num_predict": 150,
            "temperature": 0.3  # Lower temperature for consistent tool use
        }
    }

    try:
        print("\nSending request to /api/generate with tools...")
        response = requests.post(
            f"{OLLMLX_URL}/api/generate",
            json=request,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            content = result.get("response", "")

            print(f"\nResponse: {content[:300]}..." if len(content) > 300 else f"\nResponse: {content}")

            # Try to find valid JSON tool call in response
            try:
                import re
                json_match = re.search(r'\{[^{}]*"calculate"[^{}]*\}|\{[^{}]*"name"\s*:\s*"calculate"[^{}]*\}|\{[^{}]*"tool_calls"[^{}]*\}', content, re.DOTALL)
                if json_match:
                    print("\nFound tool call JSON pattern in response")
                    print("\nGenerate tool calling test: PASSED")
                    return True
            except:
                pass

            # Check if response mentions the tool or the calculation
            if "calculate" in content.lower() and ("25" in content or "17" in content or "425" in content):
                print("\nModel acknowledges tool or provides calculation")
                print("\nGenerate tool calling test: PASSED")
                return True
            else:
                print("\nNo clear tool call indication in response")
                print("\nGenerate tool calling test: FAILED")
                return False
        else:
            print(f"\nError {response.status_code}: {response.text}")
            print("\nGenerate tool calling test: FAILED")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\nCould not connect to {OLLMLX_URL}")
        print("Make sure ollmlx server is running: ./ollmlx serve")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False


def check_server():
    """Check if the ollmlx server is running."""
    try:
        response = requests.get(f"{OLLMLX_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"Server is running. Found {len(models)} models:")
            for m in models[:5]:  # Show first 5
                print(f"  - {m.get('name', 'unknown')}")
            if len(models) > 5:
                print(f"  ... and {len(models) - 5} more")
            return True
    except:
        pass

    print("Server is not running or not responding.")
    print(f"Start it with: ./ollmlx serve")
    return False


def main():
    parser = argparse.ArgumentParser(description="Test ollmlx image and tool calling features")
    parser.add_argument("--test-images", action="store_true", help="Test image support")
    parser.add_argument("--test-tools", action="store_true", help="Test tool calling")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--vision-model", default="mlx-community/llava-1.5-7b-4bit",
                       help="Vision model to use for image tests")
    parser.add_argument("--text-model", default="mlx-community/Qwen2.5-3B-Instruct-4bit",
                       help="Text model to use for tool calling tests")
    parser.add_argument("--url", default="http://localhost:11434",
                       help="ollmlx server URL")

    args = parser.parse_args()

    global OLLMLX_URL
    OLLMLX_URL = args.url

    print("=" * 60)
    print("ollmlx Feature Test Suite")
    print("=" * 60)
    print(f"Server URL: {OLLMLX_URL}")

    # Check server
    if not check_server():
        sys.exit(1)

    # Determine what to test
    test_images = args.test_images or args.all
    test_tools = args.test_tools or args.all

    # If nothing specified, run all
    if not (args.test_images or args.test_tools or args.all):
        print("\nNo specific test selected. Running all tests...")
        test_images = True
        test_tools = True

    results = []

    # Image tests
    if test_images:
        print("\n" + "#" * 60)
        print("# IMAGE SUPPORT TESTS")
        print("#" * 60)
        results.append(("Image /api/generate", test_image_support_generate(args.vision_model)))
        results.append(("Image /api/chat", test_image_support_chat(args.vision_model)))

    # Tool calling tests
    if test_tools:
        print("\n" + "#" * 60)
        print("# TOOL CALLING TESTS")
        print("#" * 60)
        results.append(("Tool calling /api/chat", test_tool_calling(args.text_model)))
        results.append(("Tool calling /api/generate", test_tool_calling_generate(args.text_model)))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
