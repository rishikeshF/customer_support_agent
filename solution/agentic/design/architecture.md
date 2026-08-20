flowchart TD
    User([user]) --> Classifier[classifier agent]
    Classifier -->|escalate to human| Escalation[escalation]
    Classifier -->|urgent| Supervisor1[Supervisor 1]
    Classifier -->|non-urgent| Supervisor2[Supervisor 2]

    Supervisor1 --> g1a
    Supervisor1 --> b1a
    Supervisor1 --> r1a
    Supervisor1 --> t1a
    Supervisor1 --> s1a

    subgraph RR["Round Robin Algorithm"]
        direction TB
        g1a[general agent 1] --> g1b[general agent 2] --> g1c[general agent 3]
        b1a[billing agent 1] --> b1b[billing agent 2] --> b1c[billing agent 3]
        r1a[reservation agent 1] --> r1b[reservation agent 2] --> r1c[reservation agent 3]
        t1a[technical agent 1] --> t1b[technical agent 2] --> t1c[technical agent 3]
        s1a[subscription agent 1] --> s1b[subscription agent 2] --> s1c[subscription agent 3]
    end

    Supervisor2 --> genAgent[general agent]
    Supervisor2 --> billAgent[billing agent]
    Supervisor2 --> resAgent[reservation agent]
    Supervisor2 --> techAgent[technical agent]
    Supervisor2 --> subAgent[subscription agent]

    g1c --> VectorDB[(VectorDB RAG)]
    b1c --> VectorDB
    r1c --> VectorDB
    t1c --> VectorDB
    s1c --> VectorDB
    g1c --> SQLiteDB[(SQLite DB)]
    b1c --> SQLiteDB
    r1c --> SQLiteDB
    t1c --> SQLiteDB
    s1c --> SQLiteDB

    genAgent --> VectorDB
    billAgent --> VectorDB
    resAgent --> VectorDB
    techAgent --> VectorDB
    subAgent --> VectorDB
    genAgent --> SQLiteDB
    billAgent --> SQLiteDB
    resAgent --> SQLiteDB
    techAgent --> SQLiteDB
    subAgent --> SQLiteDB