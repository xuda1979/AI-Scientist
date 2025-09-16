import re

# Threshold for line length (characters) before considering a formula too long
MAX_LINE_LENGTH = 80

def break_display_equation(equation):
    # Try to break at +, -, =, or , for readability
    # This is a simple heuristic; for complex equations, manual review may still be needed
    for op in [r'\\,', r'\+', r'-', r'=', r',']:
        parts = re.split(f'({op})', equation)
        if len(parts) > 1:
            # Rejoin with line breaks at the operator
            new_eq = ''
            line = ''
            for part in parts:
                if len(line) + len(part) > MAX_LINE_LENGTH:
                    new_eq += line + r'\\\n'
                    line = ''
                line += part
            new_eq += line
            return new_eq
    return equation

def insert_allowbreak_inline(math_expr):
    # Insert \allowbreak after common breakpoints in inline math
    return re.sub(r'([=+\-])', r'\1\\allowbreak ', math_expr)

def process_latex_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    in_equation = False
    equation_buffer = []
    for line in lines:
        # Detect display math environments
        if re.match(r'\\\[|\\begin\{(equation|align|multline)\}', line.strip()):
            in_equation = True
            equation_buffer = [line]
            continue
        if in_equation:
            equation_buffer.append(line)
            if re.match(r'\\\]|\\end\{(equation|align|multline)\}', line.strip()):
                # Join and process the equation
                eq_text = ''.join(equation_buffer)
                eq_body = ''.join(equation_buffer[1:-1])
                if len(eq_body) > MAX_LINE_LENGTH:
                    eq_body = break_display_equation(eq_body)
                eq_text = equation_buffer[0] + eq_body + equation_buffer[-1]
                new_lines.append(eq_text)
                in_equation = False
                equation_buffer = []
            continue
        # Process inline math
        def inline_repl(match):
            expr = match.group(1)
            if len(expr) > MAX_LINE_LENGTH:
                return f'${insert_allowbreak_inline(expr)}$'
            return match.group(0)
        line = re.sub(r'\$(.+?)\$', inline_repl, line)
        new_lines.append(line)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        # In-place modification
        input_path = sys.argv[1]
        process_latex_file(input_path, input_path)
    elif len(sys.argv) == 3:
        process_latex_file(sys.argv[1], sys.argv[2])
    else:
        print('Usage: python fix_latex_overflow.py input.tex [output.tex]')
        print('If only input.tex is given, the file is modified in place.')
        sys.exit(1)
