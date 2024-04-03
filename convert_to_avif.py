"""
Script to convert images to avif using avifenc
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    args = sys.argv[1:]
    try:
        sep_i = args.index("--")
    except ValueError:
        sep_i = None

    if sep_i is not None:
        avifenc_args = args[sep_i + 1 :]
        image_files = args[:sep_i]
    else:
        avifenc_args = []
        image_files = args

    for image_file in image_files:
        out_file = os.path.splitext(image_file)[0] + ".avif"
        print("converting", image_file, out_file)
        subprocess.run(["avifenc", image_file, out_file] + avifenc_args)
