import sys

import dotenv

dotenv.load_dotenv(override=True)

if sys.platform == "linux":
    # For Linux, `sqlite3` can be outdated, so use the installed python package.
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
