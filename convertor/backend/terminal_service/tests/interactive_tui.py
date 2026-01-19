"""
Interactive TUI Client for Terminal Service Testing.

Real-time terminal testing with:
- Multiple shell types (PowerShell, CMD, Bash via WSL)
- File creation and writing
- Live output display
- Session management

Usage: python tests/interactive_tui.py
"""

import asyncio
import httpx
import websockets
import json
import base64
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.text import Text
from datetime import datetime
import sys

console = Console()

BASE_URL = "http://localhost:8081"
WS_BASE = "ws://localhost:8081"


class TerminalTUI:
    """Interactive TUI for terminal service."""
    
    def __init__(self):
        self.sessions = {}
        self.active_outputs = {}
        
    async def create_session(self, shell_type: str):
        """Create a new terminal session."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BASE_URL}/api/v1/sessions",
                    json={
                        "shell": shell_type,
                        "rows": 24,
                        "cols": 80
                    }
                )
                
                if response.status_code == 201:
                    data = response.json()
                    session_id = data["session_id"]
                    self.sessions[session_id] = {
                        "shell": shell_type,
                        "created_at": datetime.now(),
                        "output": []
                    }
                    console.print(f"[green]✓[/green] Created {shell_type} session: {session_id[:8]}...")
                    return session_id
                else:
                    console.print(f"[red]✗ Failed:[/red] {response.text}")
                    return None
                    
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            return None
    
    async def list_sessions(self):
        """List all active sessions."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{BASE_URL}/api/v1/sessions")
                
                if response.status_code == 200:
                    sessions = response.json()
                    
                    table = Table(title="Active Terminal Sessions")
                    table.add_column("Session ID", style="cyan")
                    table.add_column("Shell", style="magenta")
                    table.add_column("PID", style="yellow")
                    table.add_column("Uptime", style="green")
                    
                    for s in sessions:
                        uptime = f"{int(s['uptime'])}s"
                        table.add_row(
                            s['session_id'][:8] + "...",
                            s['shell'],
                            str(s['pid']),
                            uptime
                        )
                    
                    console.print(table)
                    return sessions
                else:
                    console.print(f"[red]Failed to list sessions[/red]")
                    return []
                    
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            return []
    
    async def send_command(self, session_id: str, command: str):
        """Send command to terminal session via WebSocket."""
        try:
            uri = f"{WS_BASE}/ws/terminal/{session_id}"
            
            async with websockets.connect(uri) as ws:
                # Send command
                command_data = base64.b64encode((command + "\n").encode()).decode()
                await ws.send(json.dumps({
                    "type": "input",
                    "data": command_data
                }))
                
                # Collect output for 2 seconds
                console.print(f"\n[yellow]Executing:[/yellow] {command}")
                console.print("[dim]─" * 60 + "[/dim]")
                
                try:
                    output_lines = []
                    async with asyncio.timeout(2.0):
                        while True:
                            msg = await ws.recv()
                            data = json.loads(msg)
                            
                            if data.get("type") == "output":
                                output = base64.b64decode(data["data"]).decode('utf-8', errors='ignore')
                                output_lines.append(output)
                                console.print(output, end="")
                                
                except asyncio.TimeoutError:
                    pass
                
                console.print("\n[dim]─" * 60 + "[/dim]")
                return "".join(output_lines)
                
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            return ""
    
    async def delete_session(self, session_id: str):
        """Delete a terminal session."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(f"{BASE_URL}/api/v1/sessions/{session_id}")
                
                if response.status_code == 204:
                    if session_id in self.sessions:
                        shell = self.sessions[session_id]["shell"]
                        del self.sessions[session_id]
                        console.print(f"[green]✓[/green] Deleted {shell} session: {session_id[:8]}...")
                        return True
                    return True
                else:
                    console.print(f"[red]✗ Failed to delete session[/red]")
                    return False
                    
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            return False
    
    async def demo_file_operations(self, session_id: str, shell_type: str):
        """Demonstrate file creation and writing."""
        console.print(f"\n[bold cyan]═══ File Operations Demo ({shell_type}) ═══[/bold cyan]\n")
        
        if shell_type == "powershell":
            commands = [
                ("Create test file", 'echo "Hello from PowerShell" > test_ps.txt'),
                ("Write more content", 'echo "Line 2" >> test_ps.txt'),
                ("Display file", 'cat test_ps.txt'),
                ("List directory", 'dir'),
            ]
        elif shell_type == "cmd":
            commands = [
                ("Create test file", 'echo Hello from CMD > test_cmd.txt'),
                ("Write more content", 'echo Line 2 >> test_cmd.txt'),
                ("Display file", 'type test_cmd.txt'),
                ("List directory", 'dir'),
            ]
        else:  # bash/wsl
            commands = [
                ("Create test file", 'echo "Hello from Bash" > test_bash.txt'),
                ("Write more content", 'echo "Line 2" >> test_bash.txt'),
                ("Display file", 'cat test_bash.txt'),
                ("List files", 'ls -la test*.txt'),
            ]
        
        for description, cmd in commands:
            console.print(f"\n[bold]{description}:[/bold]")
            await self.send_command(session_id, cmd)
            await asyncio.sleep(0.5)
    
    async def cleanup_files(self, session_id: str, shell_type: str):
        """Clean up test files."""
        console.print(f"\n[bold yellow]═══ Cleanup ({shell_type}) ═══[/bold yellow]\n")
        
        if shell_type == "powershell":
            cleanup_cmds = [
                'rm test_ps.txt -ErrorAction SilentlyContinue',
                'dir test*.txt'
            ]
        elif shell_type == "cmd":
            cleanup_cmds = [
                'del test_cmd.txt 2>nul',
                'dir test*.txt'
            ]
        else:
            cleanup_cmds = [
                'rm -f test_bash.txt',
                'ls test*.txt 2>/dev/null || echo "All cleaned up!"'
            ]
        
        for cmd in cleanup_cmds:
            await self.send_command(session_id, cmd)
            await asyncio.sleep(0.3)
    
    async def run_full_demo(self):
        """Run complete demo with all shell types."""
        console.print(Panel.fit(
            "[bold cyan]Advanced Terminal Service - Interactive Demo[/bold cyan]\n"
            "Testing multiple shell types with file operations",
            border_style="cyan"
        ))
        
        # Test configurations
        shells = [
            ("powershell", "PowerShell"),
            ("cmd", "Command Prompt"),
        ]
        
        # Check if WSL is available
        if Confirm.ask("\nDo you have WSL installed for Bash testing?", default=False):
            shells.append(("wsl", "Bash (WSL)"))
        
        session_ids = []
        
        try:
            # Phase 1: Create all sessions
            console.print("\n[bold green]Phase 1: Creating Terminal Sessions[/bold green]")
            console.print("─" * 60)
            
            for shell, name in shells:
                session_id = await self.create_session(shell)
                if session_id:
                    session_ids.append((session_id, shell, name))
                await asyncio.sleep(0.5)
            
            # Phase 2: List all sessions
            console.print("\n[bold green]Phase 2: Listing Active Sessions[/bold green]")
            console.print("─" * 60)
            await self.list_sessions()
            
            input("\nPress Enter to continue to file operations...")
            
            # Phase 3: File operations in each terminal
            console.print("\n[bold green]Phase 3: File Operations in Each Terminal[/bold green]")
            console.print("─" * 60)
            
            for session_id, shell, name in session_ids:
                await self.demo_file_operations(session_id, shell)
                input(f"\nPress Enter to continue to next terminal...")
            
            # Phase 4: Cleanup files
            console.print("\n[bold green]Phase 4: Cleaning Up Files[/bold green]")
            console.print("─" * 60)
            
            for session_id, shell, name in session_ids:
                await self.cleanup_files(session_id, shell)
                await asyncio.sleep(0.5)
            
            input("\nPress Enter to start deleting sessions...")
            
            # Phase 5: Delete sessions one by one
            console.print("\n[bold green]Phase 5: Deleting Sessions One by One[/bold green]")
            console.print("─" * 60)
            
            for session_id, shell, name in session_ids:
                if Confirm.ask(f"\nDelete {name} session?", default=True):
                    await self.delete_session(session_id)
                    
                    # Show remaining sessions
                    console.print("\n[dim]Remaining sessions:[/dim]")
                    await self.list_sessions()
                    await asyncio.sleep(0.5)
            
            console.print("\n[bold green]✓ Demo Complete![/bold green]")
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted by user[/yellow]")
            # Cleanup remaining sessions
            console.print("\nCleaning up remaining sessions...")
            for session_id, _, _ in session_ids:
                try:
                    await self.delete_session(session_id)
                except:
                    pass
        
        except Exception as e:
            console.print(f"\n[red]Error in demo: {str(e)}[/red]")
            import traceback
            traceback.print_exc()
    
    async def interactive_mode(self):
        """Interactive menu-driven mode."""
        while True:
            console.print("\n[bold cyan]═══ Terminal Service Menu ═══[/bold cyan]\n")
            console.print("1. Create PowerShell session")
            console.print("2. Create CMD session")
            console.print("3. Create Bash (WSL) session")
            console.print("4. List all sessions")
            console.print("5. Send command to session")
            console.print("6. Delete session")
            console.print("7. Run full demo")
            console.print("8. Exit")
            
            choice = Prompt.ask("\nSelect option", choices=["1", "2", "3", "4", "5", "6", "7", "8"])
            
            if choice == "1":
                await self.create_session("powershell")
            elif choice == "2":
                await self.create_session("cmd")
            elif choice == "3":
                await self.create_session("wsl")
            elif choice == "4":
                await self.list_sessions()
            elif choice == "5":
                session_id = Prompt.ask("Enter session ID")
                command = Prompt.ask("Enter command")
                await self.send_command(session_id, command)
            elif choice == "6":
                session_id = Prompt.ask("Enter session ID")
                await self.delete_session(session_id)
            elif choice == "7":
                await self.run_full_demo()
            elif choice == "8":
                console.print("\n[yellow]Goodbye![/yellow]")
                break


async def main():
    """Main entry point."""
    tui = TerminalTUI()
    
    console.print(Panel.fit(
        "[bold]Advanced Terminal Service - Interactive TUI[/bold]\n"
        f"Service URL: {BASE_URL}\n"
        f"WebSocket: {WS_BASE}",
        border_style="green"
    ))
    
    # Check if service is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/api/v1/health")
            if response.status_code == 200:
                console.print("[green]✓ Service is running[/green]\n")
            else:
                console.print("[red]✗ Service responded but health check failed[/red]")
                return
    except Exception as e:
        console.print(f"[red]✗ Cannot connect to service at {BASE_URL}[/red]")
        console.print(f"Make sure the service is running: python run.py")
        return
    
    # Choose mode
    if Confirm.ask("Run full automated demo?", default=True):
        await tui.run_full_demo()
    else:
        await tui.interactive_mode()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]")
