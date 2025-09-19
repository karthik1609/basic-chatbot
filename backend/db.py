import os
import logging
from typing import Optional

from sqlalchemy import (
    create_engine,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from .logging_setup import configure_logging


configure_logging()
logger = logging.getLogger("db")


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "postgresql+psycopg://chatbot:chatbot@localhost:5432/chatbot")
    return url


def get_engine(echo: bool = False) -> Engine:
    url = get_database_url()
    logger.info("Connecting to database", extra={"url": url})
    engine = create_engine(url, echo=echo, pool_pre_ping=True)
    return engine


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS car_catalog (
    id SERIAL PRIMARY KEY,
    make TEXT NOT NULL,
    model TEXT NOT NULL,
    year INTEGER NOT NULL,
    body_type TEXT,
    fuel_type TEXT,
    trim TEXT,
    UNIQUE (make, model, year)
);

-- Existing clients with warranty info and car foreign key
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE,
    full_name TEXT,
    car_id INTEGER REFERENCES car_catalog(id),
    package_plan TEXT,
    warranty_status TEXT,
    warranty_start TIMESTAMP,
    warranty_end TIMESTAMP,
    last_interaction_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_users_warranty_status ON users(warranty_status);
CREATE INDEX IF NOT EXISTS idx_users_package_plan ON users(package_plan);

-- Warranty claims made by existing clients
CREATE TABLE IF NOT EXISTS warranty_claims (
    claim_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    car_id INTEGER NOT NULL REFERENCES car_catalog(id) ON DELETE RESTRICT,
    opened_at TIMESTAMP NOT NULL,
    closed_at TIMESTAMP,
    status TEXT NOT NULL,
    description TEXT
);
CREATE INDEX IF NOT EXISTS idx_claims_status ON warranty_claims(status);
CREATE INDEX IF NOT EXISTS idx_claims_user_car ON warranty_claims(user_id, car_id);

-- Sales pipeline with optional car reference
CREATE TABLE IF NOT EXISTS sales_pipeline (
    id SERIAL PRIMARY KEY,
    prospect_email TEXT,
    stage TEXT NOT NULL,
    car_id INTEGER REFERENCES car_catalog(id),
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    last_activity_at TIMESTAMP,
    next_follow_up_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sales_stage ON sales_pipeline(stage);
"""


def _rand_choice(rng, items):
    return items[rng.randrange(0, len(items))]


def seed_realistic_data(engine: Optional[Engine] = None, seed: int = 42) -> None:
    import random
    from datetime import datetime, timedelta

    rng = random.Random(seed)
    engine = engine or get_engine()

    makes_models = {
        "Toyota": ["Corolla", "Camry", "RAV4", "Highlander"],
        "Honda": ["Civic", "Accord", "CR-V"],
        "Ford": ["F-150", "Escape", "Explorer"],
        "BMW": ["3 Series", "5 Series", "X3"],
        "Audi": ["A3", "A4", "Q5"],
        "Hyundai": ["Elantra", "Tucson", "Santa Fe"],
        "Kia": ["Seltos", "Sportage", "Sorento"],
        "Tesla": ["Model 3", "Model Y"],
    }
    years = list(range(2016, 2025))
    body_types = ["Sedan", "SUV", "Truck", "Hatchback"]
    fuel_types = ["Petrol", "Diesel", "Hybrid", "Electric"]
    trims = ["Base", "Premium", "Sport", "Limited"]

    package_plans = ["Basic", "Standard", "Premium", "Platinum"]
    warranty_statuses = ["active", "expired", "cancelled", "suspended"]
    claim_statuses = ["open", "in_review", "approved", "rejected", "closed"]
    sales_stages = ["lead", "qualified", "proposal", "negotiation", "won", "lost"]

    with engine.begin() as conn:
        logger.info("Clearing existing synthetic data")
        conn.exec_driver_sql("DELETE FROM warranty_claims;")
        conn.exec_driver_sql("DELETE FROM sales_pipeline;")
        conn.exec_driver_sql("DELETE FROM users;")
        conn.exec_driver_sql("DELETE FROM car_catalog;")

        # Generate 40-50 catalog rows
        rows_catalog = []
        while len(rows_catalog) < 45:
            make = _rand_choice(rng, list(makes_models.keys()))
            model = _rand_choice(rng, makes_models[make])
            year = _rand_choice(rng, years)
            body = _rand_choice(rng, body_types)
            fuel = _rand_choice(rng, fuel_types)
            trim = _rand_choice(rng, trims)
            rows_catalog.append((make, model, year, body, fuel, trim))
        # Remove duplicates while preserving order
        seen = set()
        deduped = []
        for r in rows_catalog:
            key = (r[0], r[1], r[2])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        rows_catalog = deduped[:50]

        conn.exec_driver_sql(
            "INSERT INTO car_catalog (make, model, year, body_type, fuel_type, trim) VALUES "
            + ",".join(["(%s,%s,%s,%s,%s,%s)"] * len(rows_catalog)),
            tuple(v for row in rows_catalog for v in row),
        )
        logger.info("Inserted car_catalog", extra={"rows": len(rows_catalog)})

        # Map (make,model,year) -> id
        res = conn.exec_driver_sql("SELECT id, make, model, year FROM car_catalog")
        catalog_list = [(row[0], row[1], row[2], row[3]) for row in res.fetchall()]

        # Generate ~500 users
        users_rows = []
        now = datetime.utcnow()
        for i in range(500):
            email = f"user{i+1}@example.com"
            full_name = f"User {i+1}"
            car_id, _, _, _ = _rand_choice(rng, catalog_list)
            plan = _rand_choice(rng, package_plans)
            # warranty start in past 5 years
            start_days_ago = rng.randint(0, 5 * 365)
            start = now - timedelta(days=start_days_ago)
            # duration 1-5 years
            duration_days = rng.randint(365, 5 * 365)
            end = start + timedelta(days=duration_days)
            status = "active" if end > now else _rand_choice(rng, ["expired", "cancelled"]) 
            last_interaction = now - timedelta(days=rng.randint(0, 120))
            users_rows.append((email, full_name, car_id, plan, status, start, end, last_interaction))

        conn.exec_driver_sql(
            "INSERT INTO users (email, full_name, car_id, package_plan, warranty_status, warranty_start, warranty_end, last_interaction_at) VALUES "
            + ",".join(["(%s,%s,%s,%s,%s,%s,%s,%s)"] * len(users_rows)),
            tuple(v for row in users_rows for v in row),
        )
        logger.info("Inserted users", extra={"rows": len(users_rows)})

        # Fetch user ids and their car
        res = conn.exec_driver_sql("SELECT id, car_id, warranty_start, warranty_end FROM users")
        users_list = [
            {
                "id": row[0],
                "car_id": row[1],
                "warranty_start": row[2],
                "warranty_end": row[3],
            }
            for row in res.fetchall()
        ]

        # Generate ~1000 warranty claims across users
        claim_rows = []
        for _ in range(1000):
            u = _rand_choice(rng, users_list)
            start = u["warranty_start"] or now - timedelta(days=365)
            end = u["warranty_end"] or now
            if end < start:
                end = start + timedelta(days=30)
            # opened within warranty window +/- 60 days
            open_time = start + timedelta(seconds=rng.randint(0, int((end - start).total_seconds()) + 60*60*24*60))
            status = _rand_choice(rng, claim_statuses)
            closed_time = None
            if status in ("approved", "rejected", "closed"):
                closed_time = open_time + timedelta(days=rng.randint(1, 60))
            description = _rand_choice(rng, [
                "Engine noise investigation",
                "Electrical system diagnostics",
                "Brake pad replacement",
                "Infotainment unit failure",
                "Air conditioning issue",
            ])
            claim_rows.append((u["id"], u["car_id"], open_time, closed_time, status, description))

        # batch insert claims
        conn.exec_driver_sql(
            "INSERT INTO warranty_claims (user_id, car_id, opened_at, closed_at, status, description) VALUES "
            + ",".join(["(%s,%s,%s,%s,%s,%s)"] * len(claim_rows)),
            tuple(v for row in claim_rows for v in row),
        )
        logger.info("Inserted warranty_claims", extra={"rows": len(claim_rows)})

        # Generate ~100 sales pipeline entries
        sales_rows = []
        for i in range(100):
            email = f"prospect{i+1}@example.com"
            stage = _rand_choice(rng, sales_stages)
            car_ref = _rand_choice(rng, [None] * 3 + [c[0] for c in catalog_list])  # allow missing car
            notes = _rand_choice(rng, [
                "Requested brochure",
                "Asked about financing",
                "Test drive scheduled",
                "Negotiating price",
                "Considering competitor",
            ])
            created = now - timedelta(days=rng.randint(0, 180))
            last_activity = created + timedelta(days=rng.randint(0, 60))
            follow_up = last_activity + timedelta(days=rng.randint(3, 30)) if stage not in ("won", "lost") else None
            sales_rows.append((email, stage, car_ref, notes, created, last_activity, follow_up))

        conn.exec_driver_sql(
            "INSERT INTO sales_pipeline (prospect_email, stage, car_id, notes, created_at, last_activity_at, next_follow_up_at) VALUES "
            + ",".join(["(%s,%s,%s,%s,%s,%s,%s)"] * len(sales_rows)),
            tuple(v for row in sales_rows for v in row),
        )
        logger.info("Inserted sales_pipeline", extra={"rows": len(sales_rows)})


def init_schema(engine: Optional[Engine] = None) -> None:
    engine = engine or get_engine()
    with engine.begin() as conn:
        logger.info("Creating schema")
        conn.exec_driver_sql(SCHEMA_SQL)
        logger.info("Schema ensured")


def alembic_revision_message() -> str:
    """Return a human-readable message describing current schema for Alembic revisions."""
    return (
        "Initial warranty schema: car_catalog, users, warranty_claims, sales_pipeline with helpful indexes"
    )


def seed_data(engine: Optional[Engine] = None) -> None:
    seed_realistic_data(engine=engine)


def healthcheck(engine: Optional[Engine] = None) -> bool:
    engine = engine or get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            return True
    except OperationalError:
        return False


