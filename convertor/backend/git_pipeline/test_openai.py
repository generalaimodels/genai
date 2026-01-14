import asyncio
import httpx
import sys
from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console(force_terminal=True, width=120, color_system="standard")
BASE_URL = "http://localhost:8000/api/git"
REPO_URL = "https://github.com/openai/openai-agents-python.git"
TARGET_FILE = "docs/sessions/advanced_sqlite_session.md"

async def test_openai_repo():
    console.print(f"[bold cyan]🚀 Testing OpenAI Agents Repo:[/bold cyan] {REPO_URL}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Submit Job
        console.print("[yellow]Submitting repository...[/yellow]")
        resp = await client.post(f"{BASE_URL}/process", json={"url": REPO_URL, "depth": 1})
        
        if resp.status_code != 200:
            console.print(f"[bold red]Failed to submit:[/bold red] {resp.text}")
            return

        data = resp.json()
        if not data['success']:
            console.print(f"[bold red]API Error:[/bold red] {data['error']}")
            return
            
        job_id = data['data']['job_id']
        console.print(f"[green]✓ Job Started. ID:[/green] {job_id}")
        
        # 2. Poll Status
        console.print("[yellow]Polling status...[/yellow]")
        status = "pending"
        
        with Live(console=console, refresh_per_second=4) as live:
            while True:
                resp = await client.get(f"{BASE_URL}/{job_id}/status")
                if resp.status_code != 200: 
                    continue
                    
                repo = resp.json()['data']
                status = repo['status']
                
                table = Table(title=f"Repo Status: {repo['name']}")
                table.add_column("Status", style="magenta")
                table.add_column("Processed", style="green")
                table.add_column("Total Files", style="blue")
                table.add_row(status, str(repo['processed_files']), str(repo['total_files']))
                live.update(table)
                
                if status in ['completed', 'failed']:
                    break
                await asyncio.sleep(0.5)

        if status == 'failed':
            console.print("[bold red]❌ Job Failed[/bold red]")
            return

        # 3. Fetch Specific Document
        console.print(f"[yellow]Fetching target file: {TARGET_FILE}[/yellow]")
        resp = await client.get(f"{BASE_URL}/{job_id}/doc", params={"path": TARGET_FILE})
        
        if resp.status_code == 404:
            console.print(f"[bold red]❌ File not found:[/bold red] {TARGET_FILE}")
            # List files
            resp_tree = await client.get(f"{BASE_URL}/{job_id}/tree")
            if resp_tree.status_code == 200:
                 with open("tree.txt", "w", encoding="utf-8") as f:
                     def print_node_file(node, level=0):
                         indent = "  " * level
                         if node['type'] == 'file':
                             f.write(f"{indent}📄 {node['name']} ({node.get('size', 0)}b) - {node['path']}\n")
                         else:
                             f.write(f"{indent}📁 {node['name']}/\n")
                             for child in node.get('children', []):
                                 print_node_file(child, level + 1)
                     
                     roots = resp_tree.json()['data']
                     for root in roots:
                         print_node_file(root)
                 console.print("[green]Tree dumped to tree.txt[/green]")
            return
            
        doc_data = resp.json()
        if not doc_data['success']:
             console.print(f"[bold red]❌ Error fetching doc:[/bold red] {doc_data.get('error')}")
             return

        doc = doc_data['data']
        console.print(f"[bold green]✓ File Retrieved Successfully![/bold green]")
        console.print(f"[bold white]Title:[/bold white] {doc.get('metadata', {}).get('title', 'N/A')}")
        console.print(f"[bold white]Size:[/bold white] {doc['size_bytes']} bytes")
        console.print("-" * 40)
        console.print(doc['content'][:500] + "...")
        console.print("-" * 40)

if __name__ == "__main__":
    try:
        asyncio.run(test_openai_repo())
    except Exception as e:
        console.print(f"[bold red]Critical Script Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
