-- ============================================
-- MedAI Platform — Production Database Init
-- Enterprise-grade PostgreSQL initialization
-- ============================================
-- NOTE: All application tables are managed by Alembic migrations.
-- This script only handles: extensions, roles, permissions,
-- helper functions, and performance/security hardening.

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ============================================
-- Roles (strict separation of privilege)
-- ============================================

DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'medai_admin') THEN
      CREATE ROLE medai_admin LOGIN PASSWORD 'CHANGE_ME_STRONG_PASSWORD';
   END IF;
END
$$;

DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'medai_app') THEN
      CREATE ROLE medai_app LOGIN PASSWORD 'CHANGE_ME_STRONG_PASSWORD';
   END IF;
END
$$;

DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'medai_readonly') THEN
      CREATE ROLE medai_readonly LOGIN PASSWORD 'CHANGE_ME_STRONG_PASSWORD';
   END IF;
END
$$;

-- ============================================
-- Permissions on public schema
-- ============================================
-- All tables live in public (managed by SQLAlchemy + Alembic).

GRANT USAGE ON SCHEMA public TO medai_app;
GRANT USAGE ON SCHEMA public TO medai_readonly;

-- App role: full DML on all current + future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO medai_app;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT USAGE, SELECT ON SEQUENCES TO medai_app;

-- Read-only: SELECT on all current + future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT ON TABLES TO medai_readonly;

-- ============================================
-- Audit log protection
-- ============================================
-- After Alembic creates audit_logs, run:
--   REVOKE UPDATE, DELETE ON audit_logs FROM medai_app;
--   GRANT INSERT, SELECT ON audit_logs TO medai_app;
-- This is handled by the first migration or a post-migration hook.

-- ============================================
-- Helper Functions
-- ============================================

-- Encrypt sensitive data (PGCrypto symmetric)
CREATE OR REPLACE FUNCTION encrypt_data(data TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN encode(
        pgp_sym_encrypt(
            data,
            current_setting('app.encryption_key'),
            'compress-algo=2, cipher-algo=aes256'
        ),
        'base64'
    );
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Encryption failed: %', SQLERRM;
        RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Decrypt sensitive data
CREATE OR REPLACE FUNCTION decrypt_data(data TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN pgp_sym_decrypt(
        decode(data, 'base64'),
        current_setting('app.encryption_key')
    );
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Decryption failed: %', SQLERRM;
        RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Integrity hash (SHA-256)
CREATE OR REPLACE FUNCTION compute_integrity_hash(data TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN encode(digest(data, 'sha256'), 'hex');
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

-- Auto-update updated_at trigger function
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Archive old audit logs (called by pg_cron or backend job)
CREATE OR REPLACE FUNCTION archive_old_audit_logs(retention_days INTEGER DEFAULT 365)
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
BEGIN
    CREATE TABLE IF NOT EXISTS audit_logs_archive (LIKE audit_logs INCLUDING ALL);

    WITH moved AS (
        DELETE FROM audit_logs
        WHERE timestamp < CURRENT_TIMESTAMP - (retention_days || ' days')::INTERVAL
        RETURNING *
    )
    INSERT INTO audit_logs_archive SELECT * FROM moved;

    GET DIAGNOSTICS archived_count = ROW_COUNT;
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- Row-Level Security Templates
-- ============================================
-- Applied per table after Alembic creates them.
-- Example (run as migration or post-init):
--
-- ALTER TABLE patient_records ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE patient_records FORCE ROW LEVEL SECURITY;
--
-- CREATE POLICY patient_isolation ON patient_records
-- FOR ALL
-- USING (patient_id = current_setting('app.current_patient')::uuid)
-- WITH CHECK (patient_id = current_setting('app.current_patient')::uuid);
--
-- CREATE POLICY doctor_access ON patient_records
-- FOR SELECT
-- USING (patient_id IN (
--     SELECT patient_id FROM doctor_patient_links
--     WHERE doctor_id = current_setting('app.current_doctor')::uuid
--       AND is_active = true
-- ));

-- ============================================
-- Performance Tuning
-- ============================================

SET random_page_cost = 1.1;
SET statement_timeout = '30s';
SET idle_in_transaction_session_timeout = '60s';
SET lock_timeout = '10s';

-- ============================================
-- Security Hardening
-- ============================================

-- Grant execute on helper functions to app role
GRANT EXECUTE ON FUNCTION encrypt_data(TEXT) TO medai_app;
GRANT EXECUTE ON FUNCTION decrypt_data(TEXT) TO medai_app;
GRANT EXECUTE ON FUNCTION compute_integrity_hash(TEXT) TO medai_app;
GRANT EXECUTE ON FUNCTION update_timestamp() TO medai_app;
GRANT EXECUTE ON FUNCTION archive_old_audit_logs(INTEGER) TO medai_admin;

-- ============================================
-- Verification
-- ============================================

DO $$
BEGIN
    RAISE NOTICE '===========================================';
    RAISE NOTICE 'MedAI Database Initialization Complete';
    RAISE NOTICE 'Tables: managed by Alembic migrations';
    RAISE NOTICE 'Roles: medai_admin, medai_app, medai_readonly';
    RAISE NOTICE 'Extensions: uuid-ossp, pgcrypto, pg_stat_statements';
    RAISE NOTICE '===========================================';
END
$$;

-- ============================================
-- Backend Contract
-- ============================================
-- The backend MUST set these session variables per request:
--
--   SET LOCAL app.current_user = '<user_uuid>';
--   SET LOCAL app.current_patient = '<patient_uuid>';
--   SET LOCAL app.encryption_key = '<key>';
--
-- SET LOCAL scopes variables to the current transaction only.
