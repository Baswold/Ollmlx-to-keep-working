class Ollmlx < Formula
  desc "Run Ollama-compatible LLMs with MLX on Apple Silicon"
  homepage "https://github.com/Baswold/ollmlx"
  url "https://github.com/Baswold/ollmlx/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"  # Update after release
  license "MIT"
  head "https://github.com/Baswold/ollmlx.git", branch: "main"

  depends_on "go" => :build
  depends_on "python@3.12"
  depends_on :macos
  depends_on arch: :arm64  # MLX requires Apple Silicon

  def install
    # Build Go binaries
    system "go", "build", *std_go_args(ldflags: "-s -w"), "-o", bin/"ollmlx", "."
    system "go", "build", *std_go_args(ldflags: "-s -w"), "-o", bin/"ollama-runner", "./cmd/runner"

    # Install Python backend
    (libexec/"mlx_backend").install Dir["mlx_backend/*"]
    (libexec/"mlx_backend").install "mlx_backend/requirements.txt" if File.exist?("mlx_backend/requirements.txt")

    # Create virtual environment with MLX dependencies
    venv = libexec/"venv"
    system Formula["python@3.12"].opt_bin/"python3.12", "-m", "venv", venv
    venv_pip = venv/"bin/pip"

    # Install MLX and dependencies into venv
    system venv_pip, "install", "--upgrade", "pip"
    system venv_pip, "install",
      "mlx>=0.15.0",
      "mlx-lm>=0.19.0",
      "fastapi>=0.104.0",
      "uvicorn>=0.24.0",
      "pydantic>=2.0.0"

    # Create wrapper script that sets up the environment
    (bin/"ollmlx").unlink if (bin/"ollmlx").exist?
    (bin/"ollmlx").write <<~EOS
      #!/bin/bash
      export OLLMLX_HOME="#{prefix}"
      export OLLAMA_PYTHON="#{venv}/bin/python3"
      export OLLMLX_BACKEND="#{libexec}/mlx_backend"
      exec "#{libexec}/ollmlx" "$@"
    EOS
    (bin/"ollmlx").chmod 0755

    # Move compiled binary to libexec
    mv bin/"ollmlx.tmp", libexec/"ollmlx" if (bin/"ollmlx.tmp").exist?

    # Build the actual binary to libexec
    system "go", "build", *std_go_args(ldflags: "-s -w"), "-o", libexec/"ollmlx", "."

    # Create models directory
    (var/"ollmlx/models").mkpath
  end

  def caveats
    <<~EOS
      ollmlx has been installed with a bundled Python environment for MLX.

      To start the server:
        ollmlx serve

      To pull and run a model:
        ollmlx pull mlx-community/gemma-3-270m-4bit
        ollmlx run mlx-community/gemma-3-270m-4bit

      Models are stored in: #{var}/ollmlx/models

      The API is compatible with Ollama clients at http://localhost:11434
    EOS
  end

  service do
    run [opt_bin/"ollmlx", "serve"]
    keep_alive true
    working_dir var/"ollmlx"
    log_path var/"log/ollmlx.log"
    error_log_path var/"log/ollmlx.log"
  end

  test do
    # Test that the binary runs
    assert_match "ollmlx", shell_output("#{bin}/ollmlx --help 2>&1")

    # Test that Python venv is set up correctly
    venv_python = libexec/"venv/bin/python3"
    assert_predicate venv_python, :executable?

    # Test that MLX is importable
    system venv_python, "-c", "import mlx.core"
  end
end
