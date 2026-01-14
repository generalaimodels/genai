from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime

class JobStatus(str, Enum):
    PENDING = "pending"
    CLONING = "cloning"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class RepoRequest(BaseModel):
    url: HttpUrl = Field(..., description="HTTPS URL of the GitHub repository")
    branch: Optional[str] = Field(None, description="Specific branch to clone (default: main/master)")
    depth: int = Field(1, ge=1, description="Clone depth (shallow clone for speed)")

class ProcessingJob(BaseModel):
    job_id: str
    repo_url: str
    status: JobStatus
    progress_percentage: int = 0
    message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

class DocumentNode(BaseModel):
    path: str
    name: str
    type: str  # "file" or "directory"
    size: int
    children: Optional[List['DocumentNode']] = None

class RepoMetadata(BaseModel):
    id: str
    name: str
    description: Optional[str]
    default_branch: str
    total_files: int
    processed_files: int
    size_bytes: int
    last_synced: datetime

class ApiResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
