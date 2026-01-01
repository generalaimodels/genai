import re

# Read the file
with open('core/parser.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the typo: ${{{escaped}}}$$ should be $${escaped}$$
content = content.replace('${{{escaped}}}$$', '$${escaped}$$')
content = content.replace('${{escaped}}$$', '$${escaped}$$')

# Write back
with open('core/parser.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed parser.py")
