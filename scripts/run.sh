#!/bin/bash

# =============================================================================
# DreamMesh4D: Complete Setup and Training Pipeline
# Video-to-4D Generation with Sparse-Controlled Gaussian-Mesh Hybrid Representation
# Complete one-click solution including environment setup and training
# =============================================================================

set -e  # Exit on any error

# Suppress PyTorch warnings
export PYTHONWARNINGS="ignore"
export TORCH_WARN_ONCE=1
export CUDA_LAUNCH_BLOCKING=0

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Environment Configuration
# =============================================================================

# Environment name
CONDA_ENV_NAME="dreammesh4d"
PYTHON_VERSION="3.9"

# CUDA configuration (auto-detect system CUDA version)
CUDA_VERSION_SYSTEM=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/' || echo "")
TORCH_CUDA_VERSION=""

# Determine appropriate PyTorch CUDA version
if [ -n "$CUDA_VERSION_SYSTEM" ]; then
    if [[ "$CUDA_VERSION_SYSTEM" == 12.* ]]; then
        TORCH_CUDA_VERSION="cu121"
        print_info "Detected CUDA $CUDA_VERSION_SYSTEM, using PyTorch cu121"
    elif [[ "$CUDA_VERSION_SYSTEM" == 11.* ]]; then
        TORCH_CUDA_VERSION="cu118"
        print_info "Detected CUDA $CUDA_VERSION_SYSTEM, using PyTorch cu118"
    else
        TORCH_CUDA_VERSION="cu121"  # Default to latest
        print_warning "Unknown CUDA version $CUDA_VERSION_SYSTEM, defaulting to cu121"
    fi
else
    TORCH_CUDA_VERSION="cu121"  # Default if detection fails
    print_warning "Could not detect CUDA version, defaulting to cu121"
fi

# =============================================================================
# Configuration Parameters
# =============================================================================

# Test dataset download URL (from README.md)
TEST_DATASET_URL="https://drive.google.com/file/d/1jn18kA2FfKMnyQ6fisIn8rhBI0dr3NFk/view"
TEST_DATASET_FILE="consistent4d_data.zip"

# Data directories
DATA_ROOT="./data"
TEST_DATA_DIR="$DATA_ROOT/input"  # Fixed: sequences are in data/input/
CURRENT_SEQUENCE=""  # Will be set based on available sequences

# Input data paths - Will be determined from test data
INPUT_IMAGE=""  # Reference image (first frame from sequence)
VIDEO_FRAMES_DIR=""  # Directory containing video frames
NUM_FRAMES=0  # Number of frames in the video

# Model paths (Zero123 models will be auto-downloaded by threestudio)
ZERO123_CONFIG=""  # Will use default from config
ZERO123_MODEL=""   # Will use default from config

# Output directories
OUTPUT_ROOT="./outputs"
STATIC_OUTPUT_DIR=""  # Will be set after static training
MESH_OUTPUT_DIR=""    # Will be set after mesh export
REFINE_OUTPUT_DIR=""  # Will be set after static refine
DYNAMIC_OUTPUT_DIR="" # Will be set after dynamic training

# Training parameters (following README recommendations)
STATIC_STEPS=1000      # Increased for better quality
REFINE_STEPS=2000  
DYNAMIC_STEPS=5000

# Mesh simplification scale (higher value = more vertices, lower value = fewer vertices)
SIMPLIFY_SCALE=32

# Isosurface resolution for mesh export
ISO_RESOLUTION=256

# =============================================================================
# Environment Setup Functions
# =============================================================================

setup_conda_environment() {
    print_info "=== ENVIRONMENT SETUP: Creating Conda Environment ==="
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found. Please install Miniconda or Anaconda."
        exit 1
    fi
    
    # Source conda
    eval "$(conda shell.bash hook)"
    
    # Check if environment already exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        print_info "Environment $CONDA_ENV_NAME already exists. Activating..."
        conda activate "$CONDA_ENV_NAME"
    else
        print_info "Creating new conda environment: $CONDA_ENV_NAME"
        conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
        conda activate "$CONDA_ENV_NAME"
    fi
    
    print_success "Conda environment $CONDA_ENV_NAME is ready and activated"
}

install_pytorch() {
    print_info "=== ENVIRONMENT SETUP: Installing PyTorch ==="
    
    # Check if PyTorch is already installed with CUDA support
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local current_cuda=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
        print_info "PyTorch with CUDA $current_cuda already installed"
        
        # Check if CUDA version matches
        if [[ "$current_cuda" == "12.1" ]] && [[ "$TORCH_CUDA_VERSION" == "cu121" ]]; then
            print_success "PyTorch CUDA version matches system CUDA"
            return 0
        elif [[ "$current_cuda" == "11.8" ]] && [[ "$TORCH_CUDA_VERSION" == "cu118" ]]; then
            print_success "PyTorch CUDA version matches system CUDA"
            return 0
        else
            print_warning "PyTorch CUDA version mismatch. Reinstalling..."
            pip uninstall torch torchvision torchaudio -y
        fi
    fi
    
    print_info "Installing PyTorch 2.2.1 with CUDA support ($TORCH_CUDA_VERSION)..."
    pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/$TORCH_CUDA_VERSION
    
    # Verify installation
    if python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
        local installed_version=$(python -c "import torch; print(torch.__version__)")
        local cuda_version=$(python -c "import torch; print(torch.version.cuda)")
        print_success "PyTorch $installed_version with CUDA $cuda_version installed successfully"
    else
        print_error "PyTorch installation failed or CUDA not available"
        exit 1
    fi
}

install_basic_dependencies() {
    print_info "=== ENVIRONMENT SETUP: Installing Basic Dependencies ==="
    
    # Create requirements_basic.txt excluding CUDA extensions
    cat > requirements_basic.txt << 'EOF'
# Core ML and training libraries
lightning==2.1.0
omegaconf==2.3.0
torch-ema==0.3.0
lpips==0.1.4
opencv-python==4.8.1.78
pillow==10.0.1
imageio==2.31.5
imageio-ffmpeg==0.4.9
scikit-image==0.21.0

# 3D and graphics libraries
trimesh==3.23.5
pymeshlab==2022.2.post4
open3d==0.17.0
plyfile==0.7.4

# Transformers and diffusion models
transformers==4.35.0
diffusers==0.21.4
accelerate==0.24.1
xformers==0.0.22.post7

# Utility libraries
tqdm==4.66.1
wandb==0.15.12
gradio==3.50.0
gdown==4.7.1

# Additional dependencies for DreamMesh4D
controlnet-aux==0.0.7
invisible-watermark==0.2.0
clip-by-openai==1.1
EOF
    
    print_info "Installing basic dependencies..."
    pip install -r requirements_basic.txt
    
    print_success "Basic dependencies installed"
}

setup_cuda_environment() {
    print_info "=== ENVIRONMENT SETUP: Configuring CUDA Environment ==="
    
    # Detect CUDA installation path
    local cuda_home=""
    
    # Common CUDA installation paths
    if [ -d "/usr/local/cuda" ]; then
        cuda_home="/usr/local/cuda"
    elif [ -d "/opt/cuda" ]; then
        cuda_home="/opt/cuda"
    elif [ -n "$CUDA_HOME" ]; then
        cuda_home="$CUDA_HOME"
    else
        print_warning "CUDA installation path not found, using default /usr/local/cuda"
        cuda_home="/usr/local/cuda"
    fi
    
    # Set CUDA environment variables
    export CUDA_HOME="$cuda_home"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    
    # Set compilation flags for CUDA extensions
    export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    export FORCE_CUDA="1"
    export CUDA_LAUNCH_BLOCKING="0"
    
    print_info "CUDA environment configured:"
    print_info "  CUDA_HOME: $CUDA_HOME"
    print_info "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
    
    # Verify CUDA compiler
    if command -v nvcc &> /dev/null; then
        local nvcc_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        print_success "NVCC compiler found: version $nvcc_version"
    else
        print_error "NVCC compiler not found. Please check CUDA installation."
        exit 1
    fi
}

install_cuda_extensions() {
    print_info "=== ENVIRONMENT SETUP: Installing CUDA Extensions ==="
    
    # Install diff-gaussian-rasterization
    print_info "Installing diff-gaussian-rasterization..."
    pip install git+https://github.com/ashawkey/diff-gaussian-rasterization.git
    
    # Install simple-knn with fix for missing header
    print_info "Installing simple-knn..."
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"
    
    git clone https://github.com/camenduru/simple-knn.git
    cd simple-knn
    
    # Apply fix for missing FLT_MAX definition
    if ! grep -q "#include <float.h>" simple_knn.cu; then
        print_info "Applying fix for missing FLT_MAX definition..."
        sed -i '1i#include <float.h>' simple_knn.cu
    fi
    
    pip install .
    cd - > /dev/null
    rm -rf "$temp_dir"
    
    # Install nerfacc
    print_info "Installing nerfacc..."
    pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.2.0_cu121.html
    
    # Install additional extensions mentioned in original setup
    print_info "Installing additional CUDA extensions..."
    
    # nvdiffrast (if needed)
    if ! python -c "import nvdiffrast" 2>/dev/null; then
        print_info "Installing nvdiffrast..."
        pip install git+https://github.com/NVlabs/nvdiffrast.git
    fi
    
    print_success "CUDA extensions installed successfully"
    
    # Verify installations
    print_info "Verifying CUDA extensions..."
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')

try:
    import diff_gaussian_rasterization
    print('✓ diff-gaussian-rasterization: OK')
except ImportError as e:
    print(f'✗ diff-gaussian-rasterization: {e}')

try:
    from simple_knn._C import distCUDA2
    print('✓ simple-knn: OK')
except ImportError as e:
    print(f'✗ simple-knn: {e}')

try:
    import nerfacc
    print('✓ nerfacc: OK')
except ImportError as e:
    print(f'✗ nerfacc: {e}')
"
}

setup_dreammesh4d() {
    print_info "=== ENVIRONMENT SETUP: Setting up DreamMesh4D ==="
    
    # Install threestudio if not already installed
    if [ ! -d "threestudio" ]; then
        print_info "Installing threestudio..."
        git clone https://github.com/threestudio-project/threestudio.git
        cd threestudio
        pip install -e .
        cd ..
    fi
    
    # Install custom threestudio extension if exists
    local custom_dir="custom/threestudio-dreammesh4d"
    if [ -d "$custom_dir" ]; then
        print_info "Installing custom threestudio-dreammesh4d extension..."
        cd "$custom_dir"
        if [ -f "setup.py" ]; then
            pip install -e .
        elif [ -f "requirements.txt" ]; then
            pip install -r requirements.txt
        fi
        cd - > /dev/null
    fi
    
    print_success "DreamMesh4D setup completed"
}

complete_environment_setup() {
    print_info "=== STARTING COMPLETE ENVIRONMENT SETUP ==="
    
    setup_conda_environment
    install_pytorch
    install_basic_dependencies
    setup_cuda_environment
    install_cuda_extensions
    setup_dreammesh4d
    
    print_success "=== ENVIRONMENT SETUP COMPLETED ==="
    print_info "Environment is ready for DreamMesh4D training!"
    
    # Save environment info
    cat > environment_info.txt << EOF
DreamMesh4D Environment Information
Generated: $(date)

Conda Environment: $CONDA_ENV_NAME
Python Version: $PYTHON_VERSION
PyTorch Version: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")
CUDA Version (System): $CUDA_VERSION_SYSTEM
CUDA Version (PyTorch): $(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "Not available")
CUDA Available: $(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "Unknown")

Environment Variables:
CUDA_HOME: $CUDA_HOME
TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST
FORCE_CUDA: $FORCE_CUDA
EOF
    
    print_info "Environment info saved to environment_info.txt"
}

# =============================================================================
# Helper Functions
# =============================================================================

check_file_exists() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        return 1
    fi
    return 0
}

check_dir_exists() {
    if [ ! -d "$1" ]; then
        print_error "Directory not found: $1"
        return 1
    fi
    return 0
}

create_dir_if_not_exists() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        print_info "Created directory: $1"
    fi
}

# Enhanced disk space check function
check_disk_space() {
    local required_gb=${1:-30}  # Default 30GB required
    local current_dir=$(pwd)
    local available_gb=$(df -BG "$current_dir" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    print_info "Checking disk space..."
    print_info "Available space: ${available_gb}GB, Required: ${required_gb}GB"
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        print_error "Insufficient disk space! Available: ${available_gb}GB, Required: ${required_gb}GB"
        print_warning "Please free up disk space before continuing."
        print_info "You can clean old training results with:"
        print_info "  rm -rf outputs/sugar-dynamic/"
        print_info "  rm -rf outputs/sugar-refine/"
        print_info "  rm -rf outputs/zero123-sai/"
        return 1
    fi
    
    # Enhanced warning for tight space conditions
    if [ "$available_gb" -lt $((required_gb + 20)) ]; then
        print_warning "Disk space is tight (${available_gb}GB available)."
        print_warning "Consider freeing up more space to avoid potential issues during training."
        print_info "Dynamic training can generate 20-25GB of checkpoint files."
    fi
    
    print_success "Disk space check passed!"
    return 0
}

find_latest_checkpoint() {
    local exp_dir="$1"
    local ckpt_dir="$exp_dir/ckpts"
    if [ -d "$ckpt_dir" ]; then
        local latest_ckpt=$(ls -t "$ckpt_dir"/*.ckpt 2>/dev/null | head -n1)
        echo "$latest_ckpt"
    else
        echo ""
    fi
}

# Enhanced find_latest_experiment function with better pattern matching
find_latest_experiment() {
    local pattern="$1"
    local latest_exp=""
    local latest_time=0
    
    print_info "Searching for experiment with pattern: $pattern" >&2
    
    # Handle different patterns for different stages
    local search_patterns=()
    case "$pattern" in
        "zero123")
            search_patterns=("zero123-sai" "zero123")
            ;;
        "sugar-refine")
            search_patterns=("sugar-refine")
            ;;
        "sugar-4dgen"|"sugar-dynamic")
            search_patterns=("sugar-dynamic" "sugar-4dgen")
            ;;
        *)
            search_patterns=("$pattern")
            ;;
    esac
    
    # Search through all matching patterns
    for search_pattern in "${search_patterns[@]}"; do
        for dir in "$OUTPUT_ROOT"/$search_pattern/*/; do
            if [[ -d "$dir" && -d "$dir/configs" && -d "$dir/ckpts" ]]; then
                local dir_time=$(stat -c %Y "$dir" 2>/dev/null || echo 0)
                if [[ "$dir_time" -gt "$latest_time" ]]; then
                    latest_time="$dir_time"
                    latest_exp="$dir"
                fi
            fi
        done
    done
    
    # Remove trailing slash
    latest_exp="${latest_exp%/}"
    
    if [ -n "$latest_exp" ]; then
        print_success "Found experiment: $latest_exp" >&2
    else
        print_warning "No experiment found for pattern: $pattern" >&2
    fi
    
    echo "$latest_exp"
}

# Download file from Google Drive
download_from_gdrive() {
    local file_id="$1"
    local output_file="$2"
    
    print_info "Downloading from Google Drive: $output_file"
    
    # Extract file ID from Google Drive URL
    if [[ "$file_id" == *"drive.google.com"* ]]; then
        file_id=$(echo "$file_id" | sed 's/.*\/d\/\([^\/]*\).*/\1/')
    fi
    
    # Use gdown if available, otherwise provide manual instructions
    if command -v gdown &> /dev/null; then
        gdown "https://drive.google.com/uc?id=$file_id" -O "$output_file"
    else
        print_warning "gdown not found. Installing gdown..."
        pip install gdown
        gdown "https://drive.google.com/uc?id=$file_id" -O "$output_file"
    fi
    
    if [ ! -f "$output_file" ]; then
        print_error "Failed to download $output_file"
        print_info "Please manually download from: https://drive.google.com/file/d/$file_id/view"
        print_info "Save it as: $output_file"
        exit 1
    fi
}

# =============================================================================
# Data Preparation Functions
# =============================================================================

download_test_dataset() {
    print_info "=== DATA PREPARATION: Downloading Test Dataset ==="
    
    create_dir_if_not_exists "$DATA_ROOT"
    
    local zip_path="$DATA_ROOT/$TEST_DATASET_FILE"
    
    if [ ! -f "$zip_path" ]; then
        print_info "Downloading Consistent4D test dataset..."
        # Extract file ID from the Google Drive URL
        local file_id="1jn18kA2FfKMnyQ6fisIn8rhBI0dr3NFk"
        download_from_gdrive "$file_id" "$zip_path"
    else
        print_info "Test dataset already downloaded: $zip_path"
    fi
    
    # Extract the dataset
    if [ ! -d "$TEST_DATA_DIR" ]; then
        print_info "Extracting test dataset..."
        unzip -q "$zip_path" -d "$DATA_ROOT"
        
        # Find the extracted directory (it might have a different name)
        local extracted_dir=$(find "$DATA_ROOT" -type d -name "*test*" -o -name "*data*" -o -name "*consistent*" | head -n1)
        if [ -n "$extracted_dir" ] && [ "$extracted_dir" != "$TEST_DATA_DIR" ]; then
            mv "$extracted_dir" "$TEST_DATA_DIR"
        fi
    fi
    
    if [ ! -d "$TEST_DATA_DIR" ]; then
        print_error "Failed to extract test dataset to $TEST_DATA_DIR"
        print_info "Please manually extract $zip_path to $TEST_DATA_DIR"
        exit 1
    fi
    
    print_success "Test dataset prepared at: $TEST_DATA_DIR"
}

list_available_sequences() {
    print_info "Available test sequences:"
    local sequences=()
    
    if [ -d "$TEST_DATA_DIR" ]; then
        for seq_dir in "$TEST_DATA_DIR"/*; do
            if [ -d "$seq_dir" ]; then
                local seq_name=$(basename "$seq_dir")
                local frame_count=$(ls -1 "$seq_dir"/*.png 2>/dev/null | wc -l)
                if [ "$frame_count" -gt 0 ]; then
                    sequences+=("$seq_name")
                    echo "  - $seq_name ($frame_count frames)"
                fi
            fi
        done
    fi
    
    if [ ${#sequences[@]} -eq 0 ]; then
        print_error "No valid sequences found in $TEST_DATA_DIR"
        exit 1
    fi
    
    echo "${sequences[@]}"
}

# New function to get sequences without printing info
get_available_sequences() {
    local sequences=()
    
    if [ -d "$TEST_DATA_DIR" ]; then
        for seq_dir in "$TEST_DATA_DIR"/*; do
            if [ -d "$seq_dir" ]; then
                local seq_name=$(basename "$seq_dir")
                local frame_count=$(ls -1 "$seq_dir"/*.png 2>/dev/null | wc -l)
                if [ "$frame_count" -gt 0 ]; then
                    sequences+=("$seq_name")
                fi
            fi
        done
    fi
    
    echo "${sequences[@]}"
}

select_sequence() {
    local sequence_name="$1"
    
    if [ -z "$sequence_name" ]; then
        # Auto-select first available sequence
        local sequences=($(get_available_sequences))
        if [ ${#sequences[@]} -eq 0 ]; then
            print_error "No valid sequences found in $TEST_DATA_DIR"
            exit 1
        fi
        sequence_name="${sequences[0]}"
        print_info "Auto-selecting sequence: $sequence_name"
    fi
    
    CURRENT_SEQUENCE="$sequence_name"
    VIDEO_FRAMES_DIR="$TEST_DATA_DIR/$sequence_name"
    
    if [ ! -d "$VIDEO_FRAMES_DIR" ]; then
        print_error "Sequence directory not found: $VIDEO_FRAMES_DIR"
        exit 1
    fi
    
    # Count frames and set reference image
    NUM_FRAMES=$(ls -1 "$VIDEO_FRAMES_DIR"/*.png 2>/dev/null | wc -l)
    
    if [ "$NUM_FRAMES" -eq 0 ]; then
        print_error "No PNG frames found in $VIDEO_FRAMES_DIR"
        exit 1
    fi
    
    # Use first frame as reference image
    INPUT_IMAGE=$(ls -1 "$VIDEO_FRAMES_DIR"/*.png 2>/dev/null | head -n1)
    
    print_success "Selected sequence: $sequence_name"
    print_info "Frames directory: $VIDEO_FRAMES_DIR"
    print_info "Number of frames: $NUM_FRAMES"
    print_info "Reference image: $INPUT_IMAGE"
}

# =============================================================================
# Pre-flight Checks (Updated)
# =============================================================================

pre_flight_checks() {
    print_info "Starting DreamMesh4D complete pipeline..."
    print_info "==============================================="

    # Check if running in correct environment
    if [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]]; then
        print_warning "Not in $CONDA_ENV_NAME environment. Activating..."
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV_NAME"
    fi

    # Check disk space (require 30GB free space)
    if ! check_disk_space 30; then
        exit 1
    fi

    # Check if CUDA is available
    if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
        print_error "CUDA is not available. Please check your GPU setup."
        print_info "You may need to run the 'setup' stage first."
        exit 1
    fi

    print_success "Environment check passed!"

    create_dir_if_not_exists "$OUTPUT_ROOT"
    
    print_success "Pre-flight checks completed!"
}

# =============================================================================
# Stage 1: Static 3D Model Generation
# =============================================================================

run_static_generation() {
    print_info "=== STAGE 1: Static 3D Model Generation ==="
    print_info "Using reference image: $INPUT_IMAGE"
    print_info "Generating 3D model using Stable Zero123..."
    
    # Following README.md example command
    python launch.py \
        --config configs/stable-zero123.yaml \
        --train \
        data.image_path="$INPUT_IMAGE" \
        trainer.max_steps=$STATIC_STEPS
    
    # Find the latest experiment directory
    STATIC_OUTPUT_DIR=$(find_latest_experiment "zero123")
    
    if [ -z "$STATIC_OUTPUT_DIR" ]; then
        print_error "Could not find static generation output directory"
        exit 1
    fi
    
    print_success "Static 3D model generation completed!"
    print_info "Output directory: $STATIC_OUTPUT_DIR"
}

# =============================================================================
# Stage 2: Mesh Export
# =============================================================================

run_mesh_export() {
    print_info "=== STAGE 2: Mesh Export ==="
    print_info "Exporting coarse mesh from 3D model..."
    
    local static_config="$STATIC_OUTPUT_DIR/configs/parsed.yaml"
    local static_checkpoint=$(find_latest_checkpoint "$STATIC_OUTPUT_DIR")
    
    if [ ! -f "$static_config" ]; then
        print_error "Static config not found: $static_config"
        exit 1
    fi
    
    if [ -z "$static_checkpoint" ]; then
        print_error "No checkpoint found in $STATIC_OUTPUT_DIR/ckpts/"
        exit 1
    fi
    
    # Following README.md example command with required obj format
    python launch.py \
        --config "$static_config" \
        --export \
        resume="$static_checkpoint" \
        system.exporter_type=mesh-exporter \
        system.exporter.fmt=obj \
        system.geometry.isosurface_method=mc-cpu \
        system.geometry.isosurface_resolution=$ISO_RESOLUTION
    
    # Find exported mesh
    local mesh_dir="$STATIC_OUTPUT_DIR/save"
    MESH_OUTPUT_DIR="$mesh_dir"
    
    local exported_mesh=$(find "$mesh_dir" -name "*.obj" 2>/dev/null | head -n1)
    
    if [ -z "$exported_mesh" ]; then
        print_error "Exported mesh not found in $mesh_dir"
        exit 1
    fi
    
    # Check mesh file size
    local mesh_size=$(du -h "$exported_mesh" | cut -f1)
    print_success "Mesh export completed!"
    print_info "Exported mesh: $exported_mesh (Size: $mesh_size)"
    
    # Store the mesh path for later use
    COARSE_MESH_PATH="$exported_mesh"
}

# =============================================================================
# Stage 3: Mesh Simplification
# =============================================================================

run_mesh_simplification() {
    print_info "=== STAGE 3: Mesh Simplification ==="
    print_info "Simplifying mesh to reduce computational overhead..."
    print_info "Simplification scale: $SIMPLIFY_SCALE (higher value = more vertices)"
    
    local simplified_dir="$MESH_OUTPUT_DIR/simplified"
    create_dir_if_not_exists "$simplified_dir"
    
    # Following README.md mesh simplification command
    python custom/threestudio-dreammesh4d/scripts/mesh_simplification.py \
        --mesh_path "$COARSE_MESH_PATH" \
        --scale $SIMPLIFY_SCALE \
        --output "$simplified_dir"
    
    # Find simplified mesh
    local simplified_mesh=$(find "$simplified_dir" -name "*simplified*.obj" -o -name "*.ply" 2>/dev/null | head -n1)
    
    if [ -z "$simplified_mesh" ]; then
        print_warning "Simplified mesh not found, using original mesh"
        simplified_mesh="$COARSE_MESH_PATH"
    else
        local orig_size=$(du -h "$COARSE_MESH_PATH" | cut -f1)
        local simp_size=$(du -h "$simplified_mesh" | cut -f1)
        print_info "Mesh size reduced from $orig_size to $simp_size"
    fi
    
    # Update mesh path to use simplified version
    COARSE_MESH_PATH="$simplified_mesh"
    
    print_success "Mesh simplification completed!"
    print_info "Using mesh: $COARSE_MESH_PATH"
}

# =============================================================================
# Stage 4: Static Refinement (SuGaR)
# =============================================================================

run_static_refinement() {
    print_info "=== STAGE 4: Static Refinement (SuGaR) ==="
    print_info "Attaching Gaussians and refining the mesh..."
    print_info "Using reference image: $INPUT_IMAGE"
    print_info "Binding to mesh: $COARSE_MESH_PATH"
    
    # Following README.md static refinement command
    python launch.py \
        --config custom/threestudio-dreammesh4d/configs/sugar_static_refine.yaml \
        --train \
        data.image_path="$INPUT_IMAGE" \
        system.geometry.surface_mesh_to_bind_path="$COARSE_MESH_PATH" \
        trainer.max_steps=$REFINE_STEPS
    
    # Find the latest refinement experiment directory
    REFINE_OUTPUT_DIR=$(find_latest_experiment "sugar-refine")
    
    if [ -z "$REFINE_OUTPUT_DIR" ]; then
        print_error "Could not find static refinement output directory"
        exit 1
    fi
    
    print_success "Static refinement completed!"
    print_info "Output directory: $REFINE_OUTPUT_DIR"
    
    # Check for exported refined mesh
    local refined_mesh_dir="$REFINE_OUTPUT_DIR/save"
    if [ -d "$refined_mesh_dir" ]; then
        local refined_mesh=$(find "$refined_mesh_dir" -name "*.ply" -o -name "*.obj" 2>/dev/null | head -n1)
        if [ -n "$refined_mesh" ]; then
            print_info "Refined mesh available: $refined_mesh"
        fi
    fi
}

# =============================================================================
# Stage 5: Dynamic Training
# =============================================================================

run_dynamic_training() {
    print_info "=== STAGE 5: Dynamic Training ==="
    print_info "Training dynamic 4D model from video frames..."
    print_info "Video frames directory: $VIDEO_FRAMES_DIR"
    print_info "Number of frames: $NUM_FRAMES"
    
    # Check disk space before starting
    if ! check_disk_space 25; then
        print_error "Insufficient disk space for dynamic training"
        exit 1
    fi
    
    local refine_checkpoint=$(find_latest_checkpoint "$REFINE_OUTPUT_DIR")
    
    if [ -z "$refine_checkpoint" ]; then
        print_error "No checkpoint found in $REFINE_OUTPUT_DIR/ckpts/"
        exit 1
    fi
    
    # Find refined mesh (prefer refined mesh over coarse mesh)
    local mesh_to_bind="$COARSE_MESH_PATH"
    local refined_mesh_dir="$REFINE_OUTPUT_DIR/save"
    if [ -d "$refined_mesh_dir" ]; then
        # Look for exported mesh from SuGaR refinement (usually .ply format)
        local refined_mesh=$(find "$refined_mesh_dir" -name "exported_mesh_*.ply" 2>/dev/null | head -n1)
        if [ -z "$refined_mesh" ]; then
            # Fallback to any .ply or .obj file
            refined_mesh=$(find "$refined_mesh_dir" -name "*.ply" -o -name "*.obj" 2>/dev/null | head -n1)
        fi
        if [ -n "$refined_mesh" ]; then
            mesh_to_bind="$refined_mesh"
            print_info "Using refined mesh: $mesh_to_bind"
        else
            print_warning "No refined mesh found, using simplified mesh: $mesh_to_bind"
        fi
    fi
    
    print_info "Starting dynamic training..."
    print_info "Expected training time: ~40-45 minutes"
    
    # Following README.md dynamic training command
    python launch.py \
        --config custom/threestudio-dreammesh4d/configs/sugar_dynamic_dg.yaml \
        --train \
        data.video_frames_dir="$VIDEO_FRAMES_DIR" \
        system.geometry.num_frames=$NUM_FRAMES \
        system.geometry.surface_mesh_to_bind_path="$mesh_to_bind" \
        system.weights="$refine_checkpoint" \
        trainer.max_steps=$DYNAMIC_STEPS
    
    # Find the latest dynamic experiment directory using corrected pattern
    DYNAMIC_OUTPUT_DIR=$(find_latest_experiment "sugar-dynamic")
    
    if [ -z "$DYNAMIC_OUTPUT_DIR" ]; then
        # Fallback: try to find any sugar-dynamic directory
        DYNAMIC_OUTPUT_DIR=$(find "$OUTPUT_ROOT" -maxdepth 2 -type d -name "sugar-dynamic" | head -n1)
        if [ -n "$DYNAMIC_OUTPUT_DIR" ]; then
            # Get the latest subdirectory
            DYNAMIC_OUTPUT_DIR=$(ls -td "$DYNAMIC_OUTPUT_DIR"/*/ | head -n1)
            DYNAMIC_OUTPUT_DIR="${DYNAMIC_OUTPUT_DIR%/}"
        fi
    fi
    
    if [ -z "$DYNAMIC_OUTPUT_DIR" ]; then
        print_error "Could not find dynamic training output directory"
        print_info "Please check outputs/sugar-dynamic/ manually"
        exit 1
    fi
    
    print_success "Dynamic training completed!"
    print_info "Output directory: $DYNAMIC_OUTPUT_DIR"
    
    # Verify checkpoint exists
    local dynamic_checkpoint=$(find_latest_checkpoint "$DYNAMIC_OUTPUT_DIR")
    if [ -n "$dynamic_checkpoint" ]; then
        print_success "Training checkpoint found: $dynamic_checkpoint"
    else
        print_warning "No checkpoint found in dynamic output directory"
    fi
}

# =============================================================================
# Stage 6: Final Export
# =============================================================================

run_final_export() {
    print_info "=== STAGE 6: Final Export ==="
    print_info "Exporting final 4D model..."
    
    # If DYNAMIC_OUTPUT_DIR is not set, try to find it
    if [ -z "$DYNAMIC_OUTPUT_DIR" ]; then
        DYNAMIC_OUTPUT_DIR=$(find_latest_experiment "sugar-dynamic")
    fi
    
    if [ -z "$DYNAMIC_OUTPUT_DIR" ]; then
        print_error "Dynamic training output directory not found"
        print_info "Please ensure dynamic training has completed successfully"
        exit 1
    fi
    
    local dynamic_config="$DYNAMIC_OUTPUT_DIR/configs/parsed.yaml"
    local dynamic_checkpoint=$(find_latest_checkpoint "$DYNAMIC_OUTPUT_DIR")
    
    if [ ! -f "$dynamic_config" ]; then
        print_error "Dynamic config not found: $dynamic_config"
        exit 1
    fi
    
    if [ -z "$dynamic_checkpoint" ]; then
        print_error "No checkpoint found in $DYNAMIC_OUTPUT_DIR/ckpts/"
        exit 1
    fi
    
    print_info "Using config: $dynamic_config"
    print_info "Using checkpoint: $dynamic_checkpoint"
    print_info "Starting export process..."
    
    # Following README.md export command
    python launch.py \
        --config "$dynamic_config" \
        --export \
        resume="$dynamic_checkpoint"
    
    print_success "Final export completed!"
    
    # Verify export results
    local export_dir="$DYNAMIC_OUTPUT_DIR/save"
    if [ -d "$export_dir" ]; then
        print_info "Export directory: $export_dir"
        
        # Check for extracted meshes
        local mesh_dir="$export_dir/extracted_textured_meshes"
        if [ -d "$mesh_dir" ]; then
            local mesh_count=$(ls -1 "$mesh_dir"/*.obj 2>/dev/null | wc -l)
            if [ "$mesh_count" -gt 0 ]; then
                print_success "Successfully exported $mesh_count textured mesh files!"
                print_info "Mesh files location: $mesh_dir"
                
                # Show first few mesh files as example
                print_info "Example mesh files:"
                ls -la "$mesh_dir"/*.obj | head -3 | while read line; do
                    print_info "  $line"
                done
                if [ "$mesh_count" -gt 3 ]; then
                    print_info "  ... and $((mesh_count - 3)) more files"
                fi
            else
                print_warning "No mesh files found in export directory"
            fi
        else
            print_warning "Extracted mesh directory not found"
        fi
        
        # List other export contents
        print_info "Export contents:"
        ls -la "$export_dir"
    else
        print_warning "Export directory not found: $export_dir"
    fi
}

# =============================================================================
# Complete Pipeline Function
# =============================================================================

run_complete_pipeline() {
    local sequence_name="$1"
    
    print_info "=== DREAMMESH4D COMPLETE PIPELINE ==="
    print_info "Starting complete video-to-4D generation pipeline"
    print_info "==========================================="
    
    # Check for existing pipeline processes
    local existing_processes=$(ps aux | grep "bash.*run.sh.*pipeline" | grep -v grep | wc -l)
    if [ "$existing_processes" -gt 1 ]; then
        print_warning "Detected $existing_processes pipeline processes running!"
        print_warning "This may cause resource conflicts and training issues."
        print_info "Consider stopping other pipeline processes before continuing."
        
        # Only prompt for confirmation in interactive mode
        if [[ -t 0 && -t 1 ]]; then
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_info "Pipeline cancelled by user"
                exit 0
            fi
        else
            print_warning "Running in non-interactive mode, continuing automatically..."
            sleep 2  # Brief pause to show the warning
        fi
    fi
    
    # Pre-flight checks
    pre_flight_checks
    
    # Download test dataset if needed
    if [ ! -d "$TEST_DATA_DIR" ]; then
        download_test_dataset
    fi
    
    # Select sequence
    select_sequence "$sequence_name"
    
    print_info "Pipeline Summary:"
    print_info "  Sequence: $CURRENT_SEQUENCE ($NUM_FRAMES frames)"
    print_info "  Static Steps: $STATIC_STEPS"
    print_info "  Refine Steps: $REFINE_STEPS" 
    print_info "  Dynamic Steps: $DYNAMIC_STEPS"
    print_info "  Expected Total Time: ~60-70 minutes"
    print_info ""
    
    # Stage 1: Static 3D Generation
    run_static_generation
    
    # Stage 2: Mesh Export and Simplification  
    run_mesh_export
    run_mesh_simplification
    
    # Stage 3: Static Refinement
    run_static_refinement
    
    # Stage 4: Dynamic Training
    run_dynamic_training
    
    # Stage 5: Final Export
    run_final_export
    
    # Final summary
    print_summary
    
    print_success "🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY! 🎉"
    print_info ""
    print_info "Final Results:"
    if [ -n "$DYNAMIC_OUTPUT_DIR" ]; then
        local mesh_dir="$DYNAMIC_OUTPUT_DIR/save/extracted_textured_meshes"
        if [ -d "$mesh_dir" ]; then
            local mesh_count=$(ls -1 "$mesh_dir"/*.obj 2>/dev/null | wc -l)
            print_success "✅ Generated $mesh_count textured 4D mesh files"
            print_info "📁 Location: $mesh_dir"
            print_info ""
            print_info "You can now use these mesh files for:"
            print_info "  - 3D visualization and animation"
            print_info "  - Further processing and editing"
            print_info "  - Integration into 3D applications"
        fi
    fi
}

# =============================================================================
# Main Execution Logic (Updated)
# =============================================================================

print_summary() {
    print_info "=== TRAINING SUMMARY ==="
    echo -e "${GREEN}✓${NC} Environment: $CONDA_ENV_NAME"
    if [ -n "$CURRENT_SEQUENCE" ]; then
        echo -e "${GREEN}✓${NC} Sequence: $CURRENT_SEQUENCE ($NUM_FRAMES frames)"
    fi
    if [ -n "$STATIC_OUTPUT_DIR" ]; then
        echo -e "${GREEN}✓${NC} Static 3D Generation: $STATIC_OUTPUT_DIR"
    fi
    if [ -n "$MESH_OUTPUT_DIR" ]; then
        echo -e "${GREEN}✓${NC} Mesh Export: $MESH_OUTPUT_DIR"
    fi
    if [ -n "$REFINE_OUTPUT_DIR" ]; then
        echo -e "${GREEN}✓${NC} Static Refinement: $REFINE_OUTPUT_DIR"
    fi
    if [ -n "$DYNAMIC_OUTPUT_DIR" ]; then
        echo -e "${GREEN}✓${NC} Dynamic Training: $DYNAMIC_OUTPUT_DIR"
    fi
    print_success "DreamMesh4D pipeline completed successfully!"
}

# Parse command line arguments
STAGE=${1:-"all"}
SEQUENCE_NAME=${2:-""}

# Show help
if [ "$STAGE" = "help" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "DreamMesh4D Complete Setup and Training Pipeline"
    echo "Usage: $0 [STAGE] [SEQUENCE_NAME]"
    echo ""
    echo "STAGES:"
    echo "  setup      - Setup complete environment (conda + dependencies + CUDA extensions)"
    echo "  pipeline   - Run complete one-click pipeline (recommended for new users)"
    echo "  all        - Run complete pipeline (same as pipeline)"
    echo "  download   - Download and prepare test dataset"
    echo "  list       - List available sequences in test dataset"
    echo "  static     - Run only static stages (generation + refinement)"
    echo "  dynamic    - Run only dynamic stages (requires static stages)"
    echo "  export     - Export results from latest dynamic training"
    echo "  help       - Show this help message"
    echo ""
    echo "SEQUENCE_NAME: Name of sequence to use (optional, auto-selects first if not provided)"
    echo ""
    echo "🚀 QUICK START (One-Click):"
    echo "  $0 pipeline                  # Complete pipeline with auto-selected sequence"
    echo "  $0 pipeline aurorus          # Complete pipeline with specific sequence"
    echo ""
    echo "💡 Step-by-Step Examples:"
    echo "  $0 setup                     # Setup environment from scratch"
    echo "  $0 download                  # Download test dataset only"
    echo "  $0 list                      # List available sequences"
    echo "  $0 static aurorus            # Run only static stages"
    echo "  $0 dynamic                   # Run dynamic stages (use existing static results)"
    echo ""
    echo "📋 Requirements:"
    echo "  - CUDA-capable GPU (recommended: 24GB+ VRAM)"
    echo "  - 30GB+ free disk space"
    echo "  - Conda environment management"
    echo ""
    echo "⏱️  Expected Time: ~60-70 minutes for complete pipeline"
    exit 0
fi

# Main execution
case $STAGE in
    "setup")
        complete_environment_setup
        ;;
    "download")
        pre_flight_checks
        download_test_dataset
        list_available_sequences
        ;;
    "list")
        if [ ! -d "$TEST_DATA_DIR" ]; then
            print_error "Test dataset not found. Run: $0 download"
            exit 1
        fi
        list_available_sequences
        ;;
    "pipeline")
        # New one-click pipeline option
        run_complete_pipeline "$SEQUENCE_NAME"
        ;;
    "static")
        pre_flight_checks
        
        # Download data if needed
        if [ ! -d "$TEST_DATA_DIR" ]; then
            download_test_dataset
        fi
        
        select_sequence "$SEQUENCE_NAME"
        
        run_static_generation
        run_mesh_export
        run_mesh_simplification
        run_static_refinement
        print_summary
        ;;
    "dynamic")
        pre_flight_checks
        
        # Find existing static results
        STATIC_OUTPUT_DIR=$(find_latest_experiment "zero123")
        REFINE_OUTPUT_DIR=$(find_latest_experiment "sugar-refine")
        
        if [ -z "$STATIC_OUTPUT_DIR" ] || [ -z "$REFINE_OUTPUT_DIR" ]; then
            print_error "Static stages not found. Please run static stages first."
            exit 1
        fi
        
        # Determine sequence from static results or use parameter
        if [ -n "$SEQUENCE_NAME" ]; then
            select_sequence "$SEQUENCE_NAME"
        else
            # Try to infer from existing static results
            if [ ! -d "$TEST_DATA_DIR" ]; then
                print_error "Test dataset not found and no sequence specified."
                print_info "Please run: $0 download"
                exit 1
            fi
            select_sequence ""  # Auto-select first sequence
        fi
        
        run_dynamic_training
        run_final_export
        print_summary
        ;;
    "export")
        # Export from latest dynamic training
        DYNAMIC_OUTPUT_DIR=$(find_latest_experiment "sugar-dynamic")
        if [ -z "$DYNAMIC_OUTPUT_DIR" ]; then
            print_error "Dynamic training results not found."
            exit 1
        fi
        run_final_export
        ;;
    "all"|*)
        # Check if environment exists, if not, set it up first
        if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
            print_info "Environment $CONDA_ENV_NAME not found. Setting up complete environment first..."
            complete_environment_setup
        fi
        
        # Run complete pipeline (same as pipeline option)
        run_complete_pipeline "$SEQUENCE_NAME"
        ;;
esac

print_success "Script execution completed!"
