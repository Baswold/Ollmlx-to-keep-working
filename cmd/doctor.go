package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/envconfig"
)

var doctorCmd = &cobra.Command{
	Use:   "doctor",
	Short: "Check your environment for ollmlx readiness",
	Long:  "Check your environment for ollmlx readiness: OS/Arch, Go, Python, and MLX dependencies.",
	Run:   DoctorHandler,
}

func DoctorHandler(cmd *cobra.Command, args []string) {
	fmt.Println("🩺  ollmlx Doctor")
	fmt.Println("====================")

	// 1. Check OS/Arch
	fmt.Print("Checking System...       ")
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		fmt.Println("✅ macOS (Apple Silicon)")
	} else if runtime.GOOS == "darwin" {
		fmt.Printf("⚠️  macOS (%s) - generic optimization\n", runtime.GOARCH)
	} else {
		fmt.Printf("⚠️  %s/%s - MLX is optimized for Apple Silicon\n", runtime.GOOS, runtime.GOARCH)
	}

	// 2. Check Go
	fmt.Print("Checking Go...           ")
	if goPath, err := exec.LookPath("go"); err == nil {
		out, _ := exec.Command(goPath, "version").Output()
		fmt.Printf("✅ %s", strings.TrimSpace(string(out)))
		// Parse version to warn if too old? Skipping for simplicity.
		fmt.Println()
	} else {
		fmt.Println("❌ Not found (required to build from source)")
	}

	// 3. Check Python & MLX
	fmt.Print("Checking Python...       ")
	
	// Logic matches server/routes_mlx.go
	pythonPath := "python3"
	source := "system"
	
	if p := os.Getenv("OLLAMA_PYTHON"); p != "" {
		pythonPath = p
		source = "OLLAMA_PYTHON"
	} else {
		home, err := os.UserHomeDir()
		if err == nil {
			// Priority 1: Application Support (Ollmlx.app standard)
			appSupport := filepath.Join(home, "Library", "Application Support", "Ollmlx", "venv", "bin", "python3")
			// Priority 2: Dotfile (Legacy/Dev)
			dotFile := filepath.Join(home, ".ollmlx", "venv", "bin", "python3")

			if _, err := os.Stat(appSupport); err == nil {
				pythonPath = appSupport
				source = "ApplicationSupport"
			} else if _, err := os.Stat(dotFile); err == nil {
				pythonPath = dotFile
				source = "venv"
			}
		}
	}

	// Verify Python version and MLX
	// We run a small script to check both
	checkScript := `
import sys
import importlib.util

v = sys.version_info
print(f"Python {v.major}.{v.minor}.{v.micro}")

if v < (3, 10):
    print("❌ Python 3.10+ required")
    sys.exit(1)

if importlib.util.find_spec("mlx") is None:
    print("❌ mlx not installed")
    sys.exit(1)
else:
    import mlx.core as mx
    print(f"✅ MLX installed")
`
	
	cmdOut, err := exec.Command(pythonPath, "-c", checkScript).CombinedOutput()
	output := strings.TrimSpace(string(cmdOut))
	
	if err != nil {
		fmt.Printf("❌ Error running python at %s (%s)\n", pythonPath, source)
		if len(output) > 0 {
			fmt.Println("   Output:")
			lines := strings.Split(output, "\n")
			for _, l := range lines {
				fmt.Printf("   > %s\n", l)
			}
		}
		if source == "venv" {
			fmt.Println("\n   Try reinstalling dependencies: ./scripts/install_ollmlx.sh")
		} else if source == "system" {
			fmt.Println("\n   Try installing dependencies: pip install -r mlx_backend/requirements.txt")
		}
	} else {
		// Output likely contains "Python 3.x.x\n✅ MLX installed"
		lines := strings.Split(output, "\n")
		// First line is version
		fmt.Printf("✅ %s (%s)\n", lines[0], source)
		
		// Second line is MLX status
		fmt.Print("Checking MLX...          ")
		if len(lines) > 1 {
			fmt.Println(lines[1])
		} else {
			fmt.Println("✅ Verified")
	}
	}
	// 4. Check Environment
	fmt.Print("Checking Environment...  ")
	fmt.Printf("✅ Models: %s\n", envconfig.Models())

	fmt.Println()
	fmt.Println("Use 'ollmlx serve' to start the server.")
}
