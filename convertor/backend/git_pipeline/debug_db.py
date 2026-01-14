import asyncio
import sys
import os
from pathlib import Path

# Add current dir to path to find db explicitly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import GitRepoDatabase

async def main():
    db_path = Path("./data/git_repos.db")
    db = GitRepoDatabase(db_path)
    await db.connect()
    
    # Get all repos
    async with db._db.execute("SELECT * FROM repositories") as cursor:
        repos = await cursor.fetchall()
        print("Repositories:")
        for repo in repos:
            r = dict(repo)
            print(f"- {r['name']} ({r['id']}): {r['status']} | Files: {r['processed_files']}/{r['total_files']}")

    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
