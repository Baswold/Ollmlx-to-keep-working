#!/usr/bin/env python3
"""
MLX Backend Validation Script

Quick validation that the MLX backend is properly configured and functional.
Run this before starting the ollmlx server to verify everything is set up correctly.
"""

import sys
import os
from pathlib import Path

def print_status(test_name, passed, message=""):
    """Print test result with formatting"""
    status = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {test_name}")
    if message:
        print(f"  {message}")
    return passed

def validate_python_version():
    """Check Python version is 3.10+"""
    version = sys.version_info
    required = (3, 10)
    passed = version >= required
    msg = f"Python {version.major}.{version.minor}.{version.micro}"
    if not passed:
        msg += f" (requires {required[0]}.{required[1]}+)"
    return print_status("Python Version", passed, msg)

def validate_dependencies():
    """Check all required dependencies are installed"""
    required_packages = [
        ("mlx", "mlx"),
        ("mlx_lm", "mlx-lm"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
    ]

    all_passed = True
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print_status(f"Package: {package_name}", True)
        except ImportError as e:
            print_status(f"Package: {package_name}", False, f"Not installed: {e}")
            all_passed = False

    return all_passed

def validate_mlx_functionality():
    """Test basic MLX functionality"""
    try:
        import mlx.core as mx

        # Simple MLX operation to verify it works
        x = mx.array([1.0, 2.0, 3.0])
        y = x * 2
        result = y.tolist()

        passed = result == [2.0, 4.0, 6.0]
        msg = "Basic operations work" if passed else "Operations failed"
        return print_status("MLX Functionality", passed, msg)
    except Exception as e:
        return print_status("MLX Functionality", False, f"Error: {e}")

def validate_server_imports():
    """Check that server.py can be imported"""
    try:
        # Add mlx_backend directory to path
        backend_dir = Path(__file__).parent
        sys.path.insert(0, str(backend_dir))

        # Try to import server module
        import server

        # Check key classes exist
        has_completion_request = hasattr(server, 'CompletionRequest')
        has_completion_response = hasattr(server, 'CompletionResponse')
        has_app = hasattr(server, 'app')

        if has_completion_request and has_completion_response and has_app:
            return print_status("Server Module", True, "All components present")
        else:
            missing = []
            if not has_completion_request: missing.append("CompletionRequest")
            if not has_completion_response: missing.append("CompletionResponse")
            if not has_app: missing.append("app")
            return print_status("Server Module", False, f"Missing: {', '.join(missing)}")

    except Exception as e:
        return print_status("Server Module", False, f"Import error: {e}")

def validate_model_cache_directory():
    """Check that model cache directory exists or can be created"""
    try:
        cache_dir = Path.home() / ".ollama" / "models" / "mlx"

        if cache_dir.exists():
            msg = f"Exists: {cache_dir}"
            passed = True
        else:
            # Try to create it
            cache_dir.mkdir(parents=True, exist_ok=True)
            msg = f"Created: {cache_dir}"
            passed = cache_dir.exists()

        return print_status("Model Cache Directory", passed, msg)
    except Exception as e:
        return print_status("Model Cache Directory", False, f"Error: {e}")

def validate_port_availability():
    """Check if default MLX backend port is available"""
    import socket

    default_port = 8023
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', default_port))
            return print_status("Port Availability", True, f"Port {default_port} is free")
    except OSError:
        return print_status("Port Availability", False, f"Port {default_port} is in use (this may be OK if server is running)")

def main():
    """Run all validation checks"""
    print("=" * 60)
    print("MLX Backend Validation")
    print("=" * 60)
    print()

    results = []

    print("1. Environment Checks:")
    results.append(validate_python_version())
    print()

    print("2. Dependency Checks:")
    results.append(validate_dependencies())
    print()

    print("3. MLX Framework:")
    results.append(validate_mlx_functionality())
    print()

    print("4. Server Components:")
    results.append(validate_server_imports())
    print()

    print("5. File System:")
    results.append(validate_model_cache_directory())
    print()

    print("6. Network:")
    validate_port_availability()  # Don't fail on this one
    print()

    # Summary
    print("=" * 60)
    passed_count = sum(results)
    total_count = len(results)

    if all(results):
        print(f"✅ All checks passed ({passed_count}/{total_count})")
        print("\nMLX backend is ready to use!")
        print("\nNext steps:")
        print("  1. Start the server: ./ollmlx serve")
        print("  2. Pull a model: ./ollmlx pull mlx-community/Llama-3.2-3B-Instruct-4bit")
        print("  3. Run inference: ./ollmlx run mlx-community/Llama-3.2-3B-Instruct-4bit")
        return 0
    else:
        print(f"❌ Some checks failed ({passed_count}/{total_count} passed)")
        print("\nFix the issues above before running ollmlx.")
        print("\nTo install missing dependencies:")
        print("  cd mlx_backend")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
