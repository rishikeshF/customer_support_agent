# Models

SQLAlchemy models used by the setup notebooks to create the databases.

| File | Database |
| --- | --- |
| `udahub.py` | `../core/udahub.db` |
| `cultpass.py` | `../external/cultpass.db` |

The agent itself reads these databases with plain `sqlite3`, so it does not
import these models.
