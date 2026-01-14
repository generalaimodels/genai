"""
Production CLI Test Suite for Git Pipeline
===========================================
Comprehensive testing with:
- Performance benchmarking
- Concurrent stress testing
- Cache validation
- API verification
"""

import asyncio
import httpx
import time
import sys
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()

BASE_URL = "http://localhost:8000/api/git"


@dataclass
class BenchmarkResult:
    operation: str
    latency_ms: float
    status_code: int
    success: bool
    data_size: int = 0


class GitPipelineCLI:
    """CLI test harness for Git Pipeline API."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results: list[BenchmarkResult] = []
    
    async def submit_repo(self, url: str) -> Optional[str]:
        """Submit repository and return job_id."""
        async with httpx.AsyncClient(timeout=300.0) as client:
            start = time.perf_counter()
            resp = await client.post(
                f"{self.base_url}/process",
                json={"url": url}
            )
            latency = (time.perf_counter() - start) * 1000
            
            self.results.append(BenchmarkResult(
                operation="POST /process",
                latency_ms=latency,
                status_code=resp.status_code,
                success=resp.status_code == 200
            ))
            
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", {}).get("job_id")
            return None
    
    async def poll_status(self, repo_id: str, timeout: int = 600) -> Dict[str, Any]:
        """Poll status until completion or timeout."""
        async with httpx.AsyncClient() as client:
            start = time.time()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Processing repository...", total=None)
                
                while time.time() - start < timeout:
                    resp = await client.get(f"{self.base_url}/{repo_id}/status")
                    
                    if resp.status_code != 200:
                        return {"error": f"Status check failed: {resp.status_code}"}
                    
                    data = resp.json().get("data", {})
                    status = data.get("status", "")
                    processed = data.get("processed_files", 0)
                    total = data.get("total_files", 0)
                    
                    progress.update(
                        task, 
                        description=f"[cyan]Status: {status} | Files: {processed}/{total}"
                    )
                    
                    if status == "completed":
                        return data
                    elif status == "failed":
                        return {"error": "Processing failed", **data}
                    
                    await asyncio.sleep(2)
            
            return {"error": "Timeout waiting for completion"}
    
    async def get_tree(
        self, 
        repo_id: str, 
        path: str = "", 
        depth: int = 2
    ) -> Dict[str, Any]:
        """Fetch tree with benchmarking."""
        async with httpx.AsyncClient() as client:
            start = time.perf_counter()
            resp = await client.get(
                f"{self.base_url}/{repo_id}/tree",
                params={"path": path, "depth": depth}
            )
            latency = (time.perf_counter() - start) * 1000
            
            self.results.append(BenchmarkResult(
                operation=f"GET /tree (depth={depth})",
                latency_ms=latency,
                status_code=resp.status_code,
                success=resp.status_code == 200,
                data_size=len(resp.content)
            ))
            
            return resp.json() if resp.status_code == 200 else {}
    
    async def get_document(
        self, 
        repo_id: str, 
        path: str,
        check_cache: bool = False
    ) -> Dict[str, Any]:
        """Fetch document with caching validation."""
        async with httpx.AsyncClient() as client:
            headers = {}
            
            # First request
            start = time.perf_counter()
            resp = await client.get(
                f"{self.base_url}/{repo_id}/doc",
                params={"path": path},
                headers=headers
            )
            latency = (time.perf_counter() - start) * 1000
            
            etag = resp.headers.get("ETag", "").strip('"')
            
            self.results.append(BenchmarkResult(
                operation="GET /doc (cold)",
                latency_ms=latency,
                status_code=resp.status_code,
                success=resp.status_code == 200,
                data_size=len(resp.content)
            ))
            
            # Second request with ETag
            if check_cache and etag:
                start = time.perf_counter()
                resp2 = await client.get(
                    f"{self.base_url}/{repo_id}/doc",
                    params={"path": path},
                    headers={"If-None-Match": f'"{etag}"'}
                )
                latency2 = (time.perf_counter() - start) * 1000
                
                self.results.append(BenchmarkResult(
                    operation="GET /doc (304)",
                    latency_ms=latency2,
                    status_code=resp2.status_code,
                    success=resp2.status_code == 304
                ))
            
            return resp.json() if resp.status_code == 200 else {}
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Fetch cache statistics."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/cache/stats")
            return resp.json() if resp.status_code == 200 else {}
    
    async def stress_test(
        self, 
        repo_id: str, 
        path: str, 
        concurrency: int = 10,
        requests: int = 100
    ) -> Dict[str, Any]:
        """Concurrent stress test."""
        async with httpx.AsyncClient() as client:
            semaphore = asyncio.Semaphore(concurrency)
            latencies: list[float] = []
            errors = 0
            
            async def make_request():
                nonlocal errors
                async with semaphore:
                    start = time.perf_counter()
                    try:
                        resp = await client.get(
                            f"{self.base_url}/{repo_id}/doc",
                            params={"path": path}
                        )
                        latency = (time.perf_counter() - start) * 1000
                        latencies.append(latency)
                        if resp.status_code != 200:
                            errors += 1
                    except Exception:
                        errors += 1
            
            console.print(f"[yellow]Stress test: {requests} requests, {concurrency} concurrent[/yellow]")
            
            start = time.perf_counter()
            await asyncio.gather(*[make_request() for _ in range(requests)])
            total_time = time.perf_counter() - start
            
            if latencies:
                latencies.sort()
                return {
                    "total_requests": requests,
                    "total_time_s": round(total_time, 2),
                    "requests_per_second": round(requests / total_time, 2),
                    "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
                    "p50_latency_ms": round(latencies[len(latencies) // 2], 2),
                    "p95_latency_ms": round(latencies[int(len(latencies) * 0.95)], 2),
                    "p99_latency_ms": round(latencies[int(len(latencies) * 0.99)], 2),
                    "errors": errors
                }
            return {"error": "No successful requests"}
    
    def print_results(self):
        """Print benchmark results table."""
        table = Table(title="Benchmark Results")
        table.add_column("Operation", style="cyan")
        table.add_column("Latency (ms)", justify="right", style="green")
        table.add_column("Status", justify="center")
        table.add_column("Size", justify="right")
        
        for r in self.results:
            status = "[green]✓[/green]" if r.success else "[red]✗[/red]"
            size = f"{r.data_size:,}" if r.data_size else "-"
            table.add_row(
                r.operation,
                f"{r.latency_ms:.2f}",
                status,
                size
            )
        
        console.print(table)


async def main():
    parser = argparse.ArgumentParser(description="Git Pipeline CLI Test Suite")
    parser.add_argument("--url", default="https://github.com/huggingface/transformers.git")
    parser.add_argument("--file", default="docs/source/en/quantization/compressed_tensors.md")
    parser.add_argument("--stress", action="store_true", help="Run stress test")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()
    
    cli = GitPipelineCLI()
    
    console.print(Panel.fit("[bold cyan]Git Pipeline CLI Test Suite[/bold cyan]"))
    
    # 1. Submit repo
    console.print(f"\n[bold]1. Submitting repository:[/bold] {args.url}")
    repo_id = await cli.submit_repo(args.url)
    
    if not repo_id:
        console.print("[red]Failed to submit repository[/red]")
        return 1
    
    console.print(f"   [green]Job ID: {repo_id}[/green]")
    
    # 2. Poll status
    console.print("\n[bold]2. Waiting for processing...[/bold]")
    status = await cli.poll_status(repo_id)
    
    if "error" in status:
        console.print(f"[red]Error: {status['error']}[/red]")
        return 1
    
    console.print(f"   [green]Completed: {status.get('total_files', 0)} files indexed[/green]")
    
    # 3. Fetch tree
    console.print("\n[bold]3. Fetching file tree...[/bold]")
    tree = await cli.get_tree(repo_id, depth=2)
    console.print(f"   [green]Tree nodes: {len(tree.get('data', []))}[/green]")
    
    # 4. Fetch document
    console.print(f"\n[bold]4. Fetching document:[/bold] {args.file}")
    doc = await cli.get_document(repo_id, args.file, check_cache=True)
    
    if doc.get("success"):
        content = doc.get("data", {}).get("content", "")
        console.print(f"   [green]Content length: {len(content):,} chars[/green]")
    
    # 5. Cache stats
    console.print("\n[bold]5. Cache statistics:[/bold]")
    stats = await cli.get_cache_stats()
    if stats.get("success"):
        for tier, data in stats.get("data", {}).items():
            hit_rate = data.get("hit_rate_percent", 0)
            console.print(f"   {tier}: {data.get('size', 0)}/{data.get('capacity', 0)} | Hit rate: {hit_rate}%")
    
    # 6. Stress test
    if args.stress:
        console.print("\n[bold]6. Stress test:[/bold]")
        stress_results = await cli.stress_test(
            repo_id, args.file, 
            concurrency=args.concurrency,
            requests=args.requests
        )
        for k, v in stress_results.items():
            console.print(f"   {k}: {v}")
    
    # Print benchmark summary
    console.print("\n")
    cli.print_results()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
