# External data

CultPass's own data — the product UDA-Hub plugs into.

| File | What it is |
| --- | --- |
| `cultpass.db` | Users, experiences, subscriptions, reservations. Built by `01_external_db_setup.ipynb`. |
| `cultpass_rag_articles.jsonl` | Help articles. Embedded into `../vectorstore/` by `02_rag_db_setup.ipynb`. |
| `cultpass_*.jsonl` | Source data the setup notebooks load. |
