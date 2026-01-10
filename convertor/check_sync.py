#!/usr/bin/env python
"""Check sync between data folder and database."""
import sqlite3
from pathlib import Path

# Check filesystem
data_dir = Path("data")
fs_files = set()
for ext in ['.ipynb', '.md', '.mdx', '.rst', '.rd']:
    fs_files.update(str(f.relative_to(data_dir).as_posix()) for f in data_dir.rglob(f'*{ext}'))

print(f"Filesystem: {len(fs_files)} files")

# Check database
conn = sqlite3.connect('backend/data/documents.db')
cur = conn.cursor()
cur.execute('SELECT path FROM documents')
db_files = set(row[0] for row in cur.fetchall())
print(f"Database: {len(db_files)} files")

# Compare
missing_in_db = fs_files - db_files
missing_in_fs = db_files - fs_files

print(f"\n❌ Missing in DB: {len(missing_in_db)} files")
for f in sorted(list(missing_in_db)[:5]):
    print(f"  - {f}")

print(f"\n❌ Missing in FS (stale): {len(missing_in_fs)} files")  
for f in sorted(list(missing_in_fs)[:5]):
    print(f"  - {f}")

ipynb_fs = {f for f in fs_files if f.endswith('.ipynb')}
ipynb_db = {f for f in db_files if f.endswith('.ipynb')}
print(f"\n📓 .ipynb files:")
print(f"  Filesystem: {len(ipynb_fs)}")
print(f"  Database: {len(ipynb_db)}")

conn.close()
