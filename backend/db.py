import os
import psycopg2
from psycopg2.extras import RealDictCursor

_pools: dict = {}

DOMAIN_URI_ENV = {
    "ml":         "POSTGRES_URL_ML",
    "dl":         "POSTGRES_URL_DL",
    "nlp":        "POSTGRES_URL_NLP",
    "cv":         "POSTGRES_URL_CV",
    "genai":      "POSTGRES_URL_GENAI",
    "ainews":     "POSTGRES_URL_AINEWS",
    "statistics": "POSTGRES_URL_STATISTICS",
}

_FALLBACK_URI_ENV = "POSTGRES_URL"
_FALLBACK_URI_DEFAULT = "postgresql://postgres:postgres@localhost:5432/blogboard"

def _init_db_schema(conn):
    """Ensure the 'articles' table exists in the PostgreSQL database."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS articles (
        id SERIAL PRIMARY KEY,
        domain VARCHAR(50) NOT NULL,
        slug VARCHAR(255) NOT NULL,
        title VARCHAR(255) NOT NULL,
        description TEXT,
        date VARCHAR(20) NOT NULL,
        tags TEXT[],
        read_time VARCHAR(50),
        image_url TEXT,
        content TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(domain, slug)
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_table_sql)
    conn.commit()


def get_db_connection(domain: str = "default"):
    """
    Return a psycopg2 connection for the given domain.
    In a high-load production environment, you would use a connection pool (e.g. psycopg2.pool).
    For our simple backend generator and API, we can open a connection or use SimpleConnectionPool.
    """
    env_var = DOMAIN_URI_ENV.get(domain, _FALLBACK_URI_ENV)
    uri = os.environ.get(env_var) or os.environ.get(_FALLBACK_URI_ENV, _FALLBACK_URI_DEFAULT)
    
    conn = psycopg2.connect(uri, cursor_factory=RealDictCursor)
    _init_db_schema(conn)
    return conn
