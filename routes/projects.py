import json
from pathlib import Path

from fastapi import APIRouter

from schemas import Project, ProjectCreate, ProjectUpdate

router = APIRouter(prefix="/api/projects", tags=["Projects"])

MOCK_FILE = Path(__file__).parent.parent / "mocks" / "projects.json"


def _load_mock():
    with open(MOCK_FILE) as f:
        return json.load(f)


@router.get("", response_model=list[Project], summary="List all projects")
def list_projects():
    """Returns all grading projects with their associated rules."""
    data = _load_mock()
    return data["projects"]


@router.get("/{project_id}", response_model=Project, summary="Get project by ID")
def get_project(project_id: str):
    """Returns a single project with its full grading rules configuration."""
    data = _load_mock()
    for project in data["projects"]:
        if project["id"] == project_id:
            return project
    return {"error": "Project not found"}, 404


@router.post("", response_model=Project, summary="Create a new project", status_code=201)
def create_project(body: ProjectCreate):
    """Creates a new grading project with the specified rules. Returns the created project."""
    data = _load_mock()
    project = data["projects"][0].copy()
    project["id"] = "proj_new"
    project["name"] = body.name
    return project


@router.put("/{project_id}", response_model=Project, summary="Update a project")
def update_project(project_id: str, body: ProjectUpdate):
    """Updates an existing project's name, description, or grading rules."""
    data = _load_mock()
    for project in data["projects"]:
        if project["id"] == project_id:
            updated = project.copy()
            if body.name:
                updated["name"] = body.name
            return updated
    return {"error": "Project not found"}, 404
