"""Command-line entry point for SwiftVR inference.

Kept for backward compatibility; equivalent to the ``swiftvr`` console script
(see ``swiftvr/cli.py``).

    python scripts/inference.py         --input low_quality.mp4 --output restored.mp4         --checkpoint checkpoints/ --upscale 4 --clip-len 24 --dtype bfloat16
"""

from swiftvr.cli import main


if __name__ == "__main__":
    main()
