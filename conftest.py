# pour que `import src.xxx` fonctionne quand on lance pytest depuis la racine
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
