from trulens.core import TruSession
import os
def main():
    """Main entry point for the DuckDB Text-to-SQL evaluation application."""
    # Initialize TruSession
    TRULENS_DB_PATH = "my-trulens.sqlite3"
    TRULENS_DB_URL = f"sqlite:///{os.path.abspath(TRULENS_DB_PATH)}"
    session = TruSession(database_url=TRULENS_DB_URL)
    session.reset_database()
    session.run_dashboard()

if __name__ == "__main__":
    main()