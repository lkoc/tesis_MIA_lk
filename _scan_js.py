"""
Parse the extracted JS to find the exact syntax error.
Uses a state machine to track strings, template literals, etc.
"""
from pathlib import Path

script = Path(r"c:\usr\ths_mia_fiis\tesis_MIA_lk\_script_extract.js").read_text(encoding='utf-8')
lines = script.splitlines()
print(f"Script: {len(script)} chars, {len(lines)} lines")

# State machine to scan JS
# Track: in_single_quote, in_double_quote, in_template, in_line_comment, in_block_comment
# Also count braces/brackets/parens

state = 'code'  # code | single | double | template | line_comment | block_comment
depth_brace = 0
depth_paren = 0
depth_bracket = 0
template_depth_stack = []  # for nested ${} inside template literals

i = 0
error_found = None

def line_col(pos, src):
    before = src[:pos]
    line = before.count('\n') + 1
    col = pos - before.rfind('\n')
    return line, col

while i < len(script):
    c = script[i]
    c2 = script[i:i+2] if i+1 < len(script) else c
    
    if state == 'line_comment':
        if c == '\n':
            state = 'code'
    elif state == 'block_comment':
        if c2 == '*/':
            state = 'code'
            i += 2
            continue
    elif state == 'single':
        if c == '\\':
            i += 2
            continue
        elif c == "'":
            state = 'code'
    elif state == 'double':
        if c == '\\':
            i += 2
            continue
        elif c == '"':
            state = 'code'
    elif state == 'template':
        if c == '\\':
            i += 2
            continue
        elif c == '`':
            state = 'code'
        elif c2 == '${':
            template_depth_stack.append(depth_brace)
            state = 'code'
            # Process ${ as entering code within template
            depth_brace += 1
            i += 2
            continue
    elif state == 'code':
        if c2 == '//':
            state = 'line_comment'
            i += 2
            continue
        elif c2 == '/*':
            state = 'block_comment'
            i += 2
            continue
        elif c == "'":
            state = 'single'
        elif c == '"':
            state = 'double'
        elif c == '`':
            state = 'template'
        elif c == '{':
            depth_brace += 1
        elif c == '}':
            # Check if this closes a template expression
            if template_depth_stack and depth_brace == template_depth_stack[-1] + 1:
                template_depth_stack.pop()
                state = 'template'
                depth_brace -= 1
            else:
                depth_brace -= 1
                if depth_brace < 0:
                    ln, col = line_col(i, script)
                    error_found = f"Unmatched '}}' at char {i}, line {ln}, col {col}"
                    print(error_found)
                    print(f"Context: ...{repr(script[max(0,i-100):i+100])}...")
                    break
        elif c == '(':
            depth_paren += 1
        elif c == ')':
            depth_paren -= 1
        elif c == '[':
            depth_bracket += 1
        elif c == ']':
            depth_bracket -= 1
    i += 1

print(f"\nFinal state: {state}")
print(f"Brace depth: {depth_brace}")
print(f"Paren depth: {depth_paren}")
print(f"Bracket depth: {depth_bracket}")
print(f"Template stack: {template_depth_stack}")

if not error_found:
    if depth_brace != 0:
        print(f"WARNING: Unbalanced braces! depth={depth_brace}")
    if depth_paren != 0:
        print(f"WARNING: Unbalanced parens! depth={depth_paren}")
    if depth_bracket != 0:
        print(f"WARNING: Unbalanced brackets! depth={depth_bracket}")
    if state != 'code':
        print(f"WARNING: Script ended in state '{state}' — unclosed string?")
    else:
        print("No obvious syntax errors found by brace/string tracker.")

# Count backticks
ticks_in_code = 0
i = 0
state2 = 'code'
for i, c in enumerate(script):
    if state2 == 'code' and c == '`':
        ticks_in_code += 1
        state2 = 'template'
    elif state2 == 'template' and c == '`' and (i == 0 or script[i-1] != '\\'):
        ticks_in_code += 1
        state2 = 'code'
print(f"\nBacktick count (template literal delimiters): {ticks_in_code} (should be even)")
