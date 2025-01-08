"""Append a footbibliography directive to all .rst files in a directory."""

import os
import sys

directory = sys.argv[1]

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".rst"):
            file_path = os.path.join(root, file)
            with open(file_path, "r+", encoding="utf-8") as f:
                content = f.read()
                if ".. footbibliography::" not in content:
                    # Append '.. footbibliography::' if not present
                    f.write("\n\n.. footbibliography::")
