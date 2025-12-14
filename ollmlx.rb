class Ollmlx < Formula
  desc "Ollmlx: Apple Silicon optimized LLM inference (Ollama compatible)"
  homepage "https://github.com/ollama/ollama"
  url "https://github.com/ollama/ollama/archive/refs/tags/v0.0.1.tar.gz" # Placeholder
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"

  depends_on "go" => :build
  depends_on "python@3.11"

  def install
    system "go", "build", *std_go_args(ldflags: "-s -w"), "."
    
    # Install Python backend
    pkgshare.install "mlx_backend"
    pkgshare.install "requirements.txt"
    
    # Create wrapper script that sets up python env
    (bin/"ollmlx").write_env_script libexec/"ollmlx", 
      PYTHONPATH: pkgshare/"mlx_backend"
  end

  def post_install
    # Install python dependencies
    system "pip3", "install", "-r", "#{pkgshare}/requirements.txt", "--break-system-packages"
  end

  test do
    system "#{bin}/ollmlx", "--version"
  end
end
