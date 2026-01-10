"""Quick test to verify database schema creation."""
import asyncio
from core.database import DocumentDatabase

async def test():
    db = DocumentDatabase('data/test.db')
    await db.connect()
    print('✓ Database schema created successfully')
    stats = await db.get_stats()
    print(f'✓ Database stats: {stats}')
    await db.close()

if __name__ == "__main__":
    asyncio.run(test())
