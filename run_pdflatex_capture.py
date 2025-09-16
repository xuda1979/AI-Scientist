# Save pdflatex terminal output to a file for error analysis
# Usage: python run_pdflatex_capture.py paper.tex
import subprocess
import sys
import os

def run_pdflatex_capture(texfile):
    base = os.path.splitext(texfile)[0]
    out_file = base + '.compile.log'
    # Run pdflatex with nonstopmode and capture all output
    with open(out_file, 'w', encoding='utf-8') as f:
        proc = subprocess.run([
            'pdflatex',
            '-interaction=nonstopmode',
            texfile
        ], stdout=f, stderr=subprocess.STDOUT)
    print(f'pdflatex output saved to {out_file}')
    if proc.returncode != 0:
        print('pdflatex exited with errors. See the log file for details.')
    else:
        print('pdflatex completed successfully.')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python run_pdflatex_capture.py paper.tex')
        sys.exit(1)
    run_pdflatex_capture(sys.argv[1])
