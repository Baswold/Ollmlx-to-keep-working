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
	"github.com/ollama/ollama/readline"
)

var doctorCmd = &cobra.Command{
	Use:   "doctor",
	Short: "Check your environment for ollmlx readiness",
	Long:  "Check your environment for ollmlx readiness: OS/Arch, Go, Python, and MLX dependencies.",
	Run:   DoctorHandler,
}

// Status indicators - clean text-based
func statusOK(msg string) {
	fmt.Printf("  %s[ok]%s  %s\n", readline.ColorSuccess, readline.ColorDefault, msg)
}

func statusWarn(msg string) {
	fmt.Printf("  %s[!]%s   %s\n", readline.ColorWarning, readline.ColorDefault, msg)
}

func statusErr(msg string) {
	fmt.Printf("  %s[x]%s   %s\n", readline.ColorError, readline.ColorDefault, msg)
}

func statusDim(msg string) {
	fmt.Printf("  %s[-]%s   %s%s%s\n", readline.ColorMuted, readline.ColorDefault, readline.ColorMuted, msg, readline.ColorDefault)
}

func DoctorHandler(cmd *cobra.Command, args []string) {
	fmt.Println()
	fmt.Printf("  %sollmlx%s  %sSystem Check%s\n", readline.ColorBold, readline.ColorDefault, readline.ColorMuted, readline.ColorDefault)
	fmt.Printf("  %s────────────────────────────────%s\n", readline.ColorMuted, readline.ColorDefault)
	fmt.Println()

	// 1. Check OS/Arch
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		statusOK("macOS on Apple Silicon")
	} else if runtime.GOOS == "darwin" {
		statusWarn(fmt.Sprintf("macOS (%s) — MLX works best on Apple Silicon", runtime.GOARCH))
	} else {
		statusWarn(fmt.Sprintf("%s/%s — MLX is optimized for Apple Silicon", runtime.GOOS, runtime.GOARCH))
	}

	// 2. Check Go
	if goPath, err := exec.LookPath("go"); err == nil {
		out, _ := exec.Command(goPath, "version").Output()
		version := strings.TrimPrefix(strings.TrimSpace(string(out)), "go version ")
		statusOK(version)
	} else {
		statusDim("Go not found (optional, for building)")
	}

	// 3. Check Python & MLX
	
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
	checkScript := `
import sys
import importlib.util

v = sys.version_info
print(f"Python {v.major}.{v.minor}.{v.micro}")

if v < (3, 10):
    print("FAIL: Python 3.10+ required")
    sys.exit(1)

if importlib.util.find_spec("mlx") is None:
    print("FAIL: mlx not installed")
    sys.exit(1)
else:
    import mlx.core as mx
    print("OK: MLX installed")
`

	cmdOut, err := exec.Command(pythonPath, "-c", checkScript).CombinedOutput()
	output := strings.TrimSpace(string(cmdOut))

	if err != nil {
		statusErr(fmt.Sprintf("Python at %s (%s)", pythonPath, source))
		if len(output) > 0 {
			lines := strings.Split(output, "\n")
			for _, l := range lines {
				fmt.Printf("       %s%s%s\n", readline.ColorMuted, l, readline.ColorDefault)
			}
		}
		if source == "venv" {
			fmt.Printf("\n       %sTry: ./scripts/install_ollmlx.sh%s\n", readline.ColorMuted, readline.ColorDefault)
		} else if source == "system" {
			fmt.Printf("\n       %sTry: pip install -r mlx_backend/requirements.txt%s\n", readline.ColorMuted, readline.ColorDefault)
		}
	} else {
		lines := strings.Split(output, "\n")
		statusOK(fmt.Sprintf("%s (%s)", lines[0], source))

		// MLX status
		if len(lines) > 1 && strings.HasPrefix(lines[1], "OK:") {
			statusOK("MLX installed")
		}
	}

	// 4. Check Environment
	statusOK(fmt.Sprintf("Models: %s", envconfig.Models()))

	fmt.Println()
	fmt.Printf("  %s────────────────────────────────%s\n", readline.ColorMuted, readline.ColorDefault)
	fmt.Println()
	fmt.Printf("  %sRun:%s  ollmlx serve\n", readline.ColorMuted, readline.ColorDefault)
	fmt.Println()
}
