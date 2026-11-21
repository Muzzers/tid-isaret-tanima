#!/bin/sh
# Tezi derler: pdflatex -> bibtex -> pdflatex -> pdflatex
# (Kaynakça ve içindekiler için iki kez daha çalışması gerekiyor.)
set -e
cd "$(dirname "$0")"

# Mac'te TinyTeX kurulu ama PATH'e eklenmemis olabiliyor.
if ! command -v pdflatex >/dev/null 2>&1; then
    for d in "$HOME"/Library/TinyTeX/bin/*; do PATH="$d:$PATH"; done
fi

pdflatex -interaction=nonstopmode tez.tex
bibtex tez
pdflatex -interaction=nonstopmode tez.tex
pdflatex -interaction=nonstopmode tez.tex

echo "bitti: tez.pdf"
