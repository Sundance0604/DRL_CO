from pathlib import Path
from experiment_core.storage import read_json
def templates():
    return [{"id":p.stem,"config":read_json(p)} for p in sorted((Path(__file__).parent/"templates").glob("*.json"))]
