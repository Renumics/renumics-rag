class HFImportError(Exception):
    def __init__(self) -> None:
        gpu_command = "pip install torch torchvision sentence-transformers accelerate"
        cpu_command = (
            gpu_command + " --extra-index-url https://download.pytorch.org/whl/cpu"
        )
        super().__init__(
            f"In order to Hugging Face embeddings models, install extra packages."
            f"\nFor GPU support: `{gpu_command}`.\nFor CPU support: `{cpu_command}`."
        )
