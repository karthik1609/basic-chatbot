from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_init'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
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
        )
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            """
DROP TABLE IF EXISTS sales_pipeline;
DROP TABLE IF EXISTS warranty_claims;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS car_catalog;
            """
        )
    )
