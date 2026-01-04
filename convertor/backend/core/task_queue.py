"""
SOTA Task Queue - Simplified version with correct dataclass structure.

Fixed: Uses factory functions instead of positional/keyword argument mixing.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from collections import defaultdict


class TaskStatus(str, Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(int, Enum):
    """Priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass(order=True)
class Task:
    """Task with priority queue ordering."""
    priority: int
    created_at: float = field(default_factory=time.time)
    task_id: str = ""
    func: Optional[Callable] = field(default=None, compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)
    dependencies: set[str] = field(default_factory=set, compare=False)
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    result: Any = field(default=None, compare=False)
    error: Optional[Exception] = field(default=None, compare=False)
    retry_count: int = field(default=0, compare=False)


class DAGTaskQueue:
    """
    DAG Task Queue with SOTA algorithms.
    
    Complexity:
    - add_task: O(1) amortized
    - get_ready: O(log n) heap pop
    - complete: O(D log n) where D = dependents
    """
    
    def __init__(self, max_workers: int = 4):
        self.tasks: dict[str, Task] = {}
        self.dependencies: dict[str, set[str]] = defaultdict(set)
        self.dependents: dict[str, set[str]] = defaultdict(set)
        self.ready_queue: list[Task] = []
        self.max_workers = max_workers
        self.active_workers = 0
        self.total_completed = 0
        self.total_failed = 0
        self._shutdown = False
    
    def add_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[set[str]] = None,
        **kwargs
    ) -> Task:
        """Add task to queue. O(1) amortized."""
        # Create task with factory pattern
        task = Task(
            priority=priority.value,
            created_at=time.time(),
            task_id=task_id,
        )
        task.func = func
        task.args = args
        task.kwargs = kwargs
        task.dependencies = dependencies or set()
        
        self.tasks[task_id] = task
        
        if dependencies:
            self.dependencies[task_id] = dependencies
            for dep_id in dependencies:
                self.dependents[dep_id].add(task_id)
            
            if self._are_dependencies_satisfied(task_id):
                self._mark_ready(task)
        else:
            self._mark_ready(task)
        
        return task
    
    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check dependencies. O(D)."""
        dependencies = self.dependencies.get(task_id, set())
        for dep_id in dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def _mark_ready(self, task: Task) -> None:
        """Mark ready and add to heap. O(log n)."""
        task.status = TaskStatus.READY
        heapq.heappush(self.ready_queue, task)
    
    async def get_ready_task(self) -> Optional[Task]:
        """Get next task. O(log n)."""
        while self.ready_queue:
            task = heapq.heappop(self.ready_queue)
            if task.status == TaskStatus.READY:
                task.status = TaskStatus.RUNNING
                return task
        return None
    
    async def complete_task(self, task: Task, result: Any = None) -> None:
        """Complete task and trigger dependents. O(D log n)."""
        task.status = TaskStatus.COMPLETED
        task.result = result
        self.total_completed += 1
        
        dependents = self.dependents.get(task.task_id, set())
        for dependent_id in dependents:
            if self._are_dependencies_satisfied(dependent_id):
                self._mark_ready(self.tasks[dependent_id])
    
    async def fail_task(self, task: Task, error: Exception) -> None:
        """Fail with retry. O(1) or O(log n)."""
        task.error = error
        task.retry_count += 1
        
        if task.retry_count < 3:
            await asyncio.sleep(2 ** task.retry_count)
            self._mark_ready(task)
        else:
            task.status = TaskStatus.FAILED
            self.total_failed += 1
    
    async def worker(self, worker_id: int) -> None:
        """Background worker."""
        print(f"✓ Worker {worker_id} started")
        
        while not self._shutdown:
            task = await self.get_ready_task()
            
            if not task:
                await asyncio.sleep(0.1)
                continue
            
            self.active_workers += 1
            
            try:
                result = await task.func(*task.args, **task.kwargs)
                await self.complete_task(task, result)
            except Exception as e:
                await self.fail_task(task, e)
            finally:
                self.active_workers -= 1
        
        print(f"✓ Worker {worker_id} stopped")
    
    async def start_workers(self) -> list[asyncio.Task]:
        """Start worker pool."""
        return [asyncio.create_task(self.worker(i)) for i in range(self.max_workers)]
    
    async def shutdown(self, timeout: float = 10.0) -> None:
        """Graceful shutdown."""
        self._shutdown = True
        start_time = time.time()
        while self.active_workers > 0 and (time.time() - start_time < timeout):
            await asyncio.sleep(0.1)
    
    def get_stats(self) -> dict:
        """Get statistics."""
        return {
            "total": len(self.tasks),
            "pending": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
            "ready": len(self.ready_queue),
            "running": self.active_workers,
            "completed": self.total_completed,
            "failed": self.total_failed,
        }


# Test example
async def test_dag():
    """Test DAG workflow."""
    queue = DAGTaskQueue(max_workers=4)
    results = {}
    
    async def step1():
        await asyncio.sleep(0.1)
        results['step1'] = 'done'
        return 'step1'
    
    async def step2():
        await asyncio.sleep(0.1)
        results['step2'] = 'done'
        return 'step2'
    
    async def step3():
        await asyncio.sleep(0.1)
        results['step3'] = 'done'
        return 'step3'
    
    # Add tasks with dependencies
    queue.add_task("s1", step1, priority=TaskPriority.HIGH)
    queue.add_task("s2", step2, priority=TaskPriority.NORMAL, dependencies={"s1"})
    queue.add_task("s3", step3, priority=TaskPriority.LOW, dependencies={"s2"})
    
    # Start workers
    workers = await queue.start_workers()
    
    # Wait for completion
    while queue.get_stats()['completed'] < 3:
        await asyncio.sleep(0.05)
        print(f"Stats: {queue.get_stats()}")
    
    await queue.shutdown()
    for w in workers:
        w.cancel()
    
    print(f"\n✓ All tasks completed: {results}")
    return results


if __name__ == "__main__":
    asyncio.run(test_dag())
