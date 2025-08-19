import dspy
import duckdb
import pandas as pd
from typing import  Optional
import time
from dataclasses import dataclass
from datasets import load_dataset
import re
import os
from dotenv import load_dotenv
print("Loading DSPy and TruLens for Text-to-SQL evaluation...")
# TruLens imports
from trulens.core import TruSession, Feedback
from trulens.core.schema.select import Select
from trulens.feedback import GroundTruthAgreement
from trulens.providers.litellm import LiteLLM
from trulens.apps.app import TruApp, instrument
import litellm

print("DSPy and TruLens loaded successfully.")
litellm.modify_params = True
load_dotenv()

dspy.configure(lm=dspy.LM("claude-3-5-haiku-20241022",api_key=os.getenv("api_key")))

class TextToSQLSignature(dspy.Signature):
    """Given schema and data manipulation statements convert natural language questions to DuckDB SQL queries"""

    schema_info = dspy.InputField(desc="Database schema information with table names, columns, types, and sample data")
    question = dspy.InputField(desc="Natural language question about the data")
    sql_query = dspy.OutputField(desc="Valid DuckDB SQL query that answers the question")
    explanation = dspy.OutputField(desc="Brief explanation of the SQL query logic")

class DSPyTextToSQL(dspy.Module):
    """DSPy module for text-to-SQL generation"""

    def __init__(self):
        super().__init__()
        self.generate_sql = dspy.ChainOfThought(TextToSQLSignature)

    def forward(self, schema_info: str, question: str):
        # Option 1: Direct return (most common)
        return self.generate_sql(schema_info=schema_info, question=question)

        # Option 2: If you need to modify/validate the output
        # result = self.generate_sql(schema_info=schema_info, question=question)
        # # Add any post-processing here if needed
        # cleaned_sql = self._clean_sql(result.sql_query)
        # return dspy.Prediction(
        #     sql_query=cleaned_sql,
        #     explanation=result.explanation
        # )

    # def _clean_sql(self, sql: str) -> str:
    #     """Clean up generated SQL if needed"""
    #     # Remove markdown formatting
    #     if sql.startswith("```sql"):
    #         sql = sql.replace("```sql", "").replace("```", "").strip()
    #     return sql
    
@dataclass
class SQLEvalResult:
    """Container for SQL evaluation results"""
    query: str
    generated_sql: str
    ground_truth_sql: str
    error: Optional[str] = None


class DuckDBTextToSQLApp:
    """Main application class integrating DSPy with TruLens ground truth evaluation"""

    def __init__(self,session: TruSession = None):
        super().__init__() # Call superclass constructor
        self.dspy_module = DSPyTextToSQL()
        self.conn = duckdb.connect(":memory:")
        if session is None:
            raise ValueError("A valid TruSession must be provided for initialization.")
        self.session = session
        # Load and prepare the dataset
        self.dataset = self._load_and_prepare_dataset()

        # Setup ground truth in TruLens
        self._setup_ground_truth_dataset()

        # Setup feedback functions
        self._setup_feedback_functions()

    def _load_and_prepare_dataset(self, num_samples: int = 100) -> pd.DataFrame:
        """Load the motherduckdb/duckdb-text2sql-25k dataset"""
        print("Loading DuckDB text2sql dataset...")

        # Load dataset from HuggingFace
        dataset = load_dataset("motherduckdb/duckdb-text2sql-25k", split="train")

        # Convert to pandas and take a subset for experimentation
        df = dataset.to_pandas().head(num_samples)

        print(f"Loaded {len(df)} samples from the dataset")
        return df

    def _prepare_ground_truth_dataframe(self) -> pd.DataFrame:
        """Prepare ground truth dataframe in TruLens format"""
        ground_truth_data = []

        for idx, row in self.dataset.iterrows():
            # Extract information from the dataset
            question = row.get('prompt', '')
            sql_query = row.get('query', '')
            schema_info = row.get('schema', '')

            # Create ground truth entry
            gt_entry = {
                'query_id': str(idx),
                'query': question,
                'expected_response': sql_query,  # Ground truth SQL
                'expected_chunks': [
                    {
                        'text': schema_info,
                        'title': 'Database Schema',
                        'expected_score': 1.0
                    }
                ],
                # Additional metadata for our use case
                'schema_info': schema_info,
                'ground_truth_sql': sql_query
            }
            ground_truth_data.append(gt_entry)

        return pd.DataFrame(ground_truth_data)

    def _setup_ground_truth_dataset(self):
        """Setup ground truth dataset in TruLens"""
        print("Setting up ground truth dataset in TruLens...")

        # Reset database for clean start
        # self.session.reset_database()

        # Prepare ground truth dataframe
        self.ground_truth_df = self._prepare_ground_truth_dataframe()

        # Add to TruLens session
        self.session.add_ground_truth_to_dataset(
            dataset_name="duckdb_text2sql_groundtruth",
            ground_truth_df=self.ground_truth_df,
            dataset_metadata={
                "domain": "Text-to-SQL",
                "dataset_source": "motherduckdb/duckdb-text2sql-25k",
                "description": "Ground truth evaluation for DuckDB text-to-SQL generation"
            }
        )

        print(f"Added {len(self.ground_truth_df)} ground truth samples to TruLens")

    def _setup_feedback_functions(self):
        """Setup TruLens feedback functions for ground truth evaluation"""

        # Initialize Claude provider for TruLens
        provider = LiteLLM(model_engine="claude-3-5-haiku-20241022")
        
        if not self._test_provider_connection(provider):
            print("❌ Provider failed, using dummy feedbacks")
            self.feedback_functions = []
            return
        # Get ground truth data for feedback functions
        gt_df = self.session.get_ground_truth("duckdb_text2sql_groundtruth")

        # Define selectors based on our application structure
        query_selector = Select.RecordCalls.generate_sql_with_evaluation.args.question
        generated_sql_selector = Select.RecordCalls.generate_sql_with_evaluation.rets.generated_sql
        schema_selector = Select.RecordCalls.generate_sql_with_evaluation.args.schema_info

        # Ground truth SQL agreement (semantic similarity)
        self.f_sql_groundtruth = Feedback(
                GroundTruthAgreement(gt_df, provider=provider).agreement_measure,
                name="SQL Ground Truth Agreement"
            ).on(query_selector).on(generated_sql_selector)
        
        # Custom SQL correctness feedback
        self.f_sql_correctness = (
            Feedback(
                self._evaluate_sql_correctness,
                name="SQL Correctness"
            )
            .on(query_selector)
            .on(generated_sql_selector)
        )
        # Custom SQL function to evaluate if sql contains valid filtering syntax
        self.f_sql_filtering = (
            Feedback(
                lambda question, generated_sql: 1.0 if "WHERE" in generated_sql.upper() else 0.0,
                name="SQL Filtering Validity"
            )
            .on(query_selector)
            .on(generated_sql_selector)
        )
        # Schema relevance feedback
        self.f_schema_relevance = (
            Feedback(
                provider.relevance,
                name="Schema Relevance"
            )
            .on(schema_selector)
            .on(generated_sql_selector)
        )
        self.feedback_functions = [
            self.f_sql_groundtruth,
            self.f_sql_correctness,
            self.f_schema_relevance
        ]
    def _evaluate_sql_correctness(self, question: str, generated_sql: str) -> float:
        """Custom feedback function to evaluate SQL correctness"""
        try:
            # Find ground truth for this question
            gt_row = self.ground_truth_df[self.ground_truth_df['query'] == question]
            if gt_row.empty:
                return 0.0

            ground_truth_sql = gt_row.iloc[0]['ground_truth_sql']
            schema_info = gt_row.iloc[0]['schema_info']

            # Basic syntax check
            if not self._is_valid_sql_syntax(generated_sql,schema_info):
                return 0.0
            # Structural similarity check (basic)
            similarity_score = self._calculate_sql_similarity(generated_sql, ground_truth_sql)

            return similarity_score

        except Exception as e:
            print(f"Error in SQL correctness evaluation: {e}")
            return 0.0

    def _is_valid_sql_syntax(self, sql: str,schema_info: str = "") -> bool:
        """Check if SQL has valid syntax"""
        try:
            # Create a temporary connection to test syntax
            test_conn = duckdb.connect(":memory:")
            # Use EXPLAIN to check syntax without executing
            self._create_tables_from_schema(test_conn, schema_info)
            test_conn.execute(f"EXPLAIN {sql}")
            test_conn.close()
            return True
        except:
            return False

    def _calculate_sql_similarity(self, sql1: str, sql2: str) -> float:
        """Calculate basic structural similarity between two SQL queries"""
        # Normalize SQL strings
        sql1_norm = self._normalize_sql(sql1)
        sql2_norm = self._normalize_sql(sql2)

        # Simple token-based similarity
        tokens1 = set(sql1_norm.split())
        tokens2 = set(sql2_norm.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union) if union else 0.0

    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL string for comparison"""
        # Remove extra whitespace and convert to lowercase
        sql = re.sub(r'\s+', ' ', sql.strip().lower())
        # Remove common SQL formatting
        sql = sql.replace('(', ' ( ').replace(')', ' ) ')
        sql = sql.replace(',', ' , ')
        return sql
    
    def _create_tables_from_schema(self, conn: duckdb.DuckDBPyConnection, schema_info: str):
        """Create tables from schema information"""
        try:
            # Split schema_info into individual CREATE statements
            # Handle different schema formats from the dataset
            print(f""" 
              CREATE TABLE in schema info : {"CREATE TABLE" in schema_info.upper()}
                  """)
            
            if "CREATE TABLE" in schema_info.upper():
                # Direct CREATE TABLE statements
                statements = self._extract_create_statements(schema_info)
                for stmt in statements:
                    if stmt.strip():
                        conn.execute(stmt)
            else:
                # Handle other schema formats (e.g., table descriptions)
                # This is a fallback for schemas that aren't in CREATE TABLE format
                self._create_dummy_tables_from_description(conn, schema_info)
                
        except Exception as e:
            # If schema creation fails, continue without it
            # The SQL validation will still catch basic syntax errors
            print(f"Warning: Could not create tables from schema: {str(e)}")
    
    def _extract_create_statements(self, schema_info: str) -> list:
        """Extract individual CREATE TABLE statements from schema"""
        import re
        
        # Find all CREATE TABLE statements (case insensitive)
        pattern = r'CREATE\s+TABLE\s+[^;]+;'
        statements = re.findall(pattern, schema_info, re.IGNORECASE | re.DOTALL)
        
        # If no semicolons, try to split on CREATE TABLE
        if not statements:
            parts = re.split(r'CREATE\s+TABLE', schema_info, flags=re.IGNORECASE)
            statements = []
            for i, part in enumerate(parts):
                if i == 0:
                    continue  # Skip the first empty part
                stmt = f"CREATE TABLE{part}"
                if not stmt.strip().endswith(';'):
                    stmt += ';'
                statements.append(stmt)
        
        return statements
    
    def _create_dummy_tables_from_description(self, conn: duckdb.DuckDBPyConnection, schema_info: str):
        """Create dummy tables when schema is in description format"""
        # This is a fallback for non-standard schema formats
        # Create a simple dummy table to allow basic SQL validation
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dummy_table (
                    id INTEGER,
                    name VARCHAR,
                    value DECIMAL
                )
            """)
        except:
            pass
    def _test_provider_connection(self,provider: LiteLLM = None) -> Optional[LiteLLM]:
        """Test if the provider is working correctly"""
        print("🔧 Testing provider connection...")
        
        try:
            # Test basic relevance call
            test_result = provider.relevance(
                prompt="What is the capital of France?",
                response="Paris is the capital of France."
            )
            
            print(f"✅ Provider test successful: {test_result}")
            return True
        
        except Exception as e:
            print(f"❌ Provider test failed: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False
    @instrument
    def generate_sql_with_evaluation(self, question: str, schema_info: str) -> SQLEvalResult:
        """Main method that generates SQL and returns evaluation results"""
        try:
            # Generate SQL using DSPy
            prediction = self.dspy_module(schema_info=schema_info, question=question)

            generated_sql = prediction.sql_query.strip()

            # Clean up SQL if wrapped in markdown
            if generated_sql.startswith("```sql"):
                generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()

            # Find ground truth SQL for comparison
            gt_row = self.ground_truth_df[self.ground_truth_df['query'] == question]
            ground_truth_sql = gt_row.iloc[0]['ground_truth_sql'] if not gt_row.empty else ""

            return SQLEvalResult(
                query=question,
                generated_sql=generated_sql,
                ground_truth_sql=ground_truth_sql,
            )

        except Exception as e:
            return SQLEvalResult(
                query=question,
                generated_sql="",
                ground_truth_sql="",
                error=str(e)
            )

    def run_evaluation_experiment(self, num_samples: int = 10,tru_app: TruApp = None):
        """Run evaluation experiment on a subset of the dataset"""

        print(f"Running evaluation experiment on {num_samples} samples...")
        print(f"Creating TruApp with {len(self.feedback_functions)} feedback functions")
        for i, f in enumerate(self.feedback_functions):
            print(f"Feedback {i+1}: {f.name}")

        # Sample from the dataset
        test_samples = self.ground_truth_df.head(num_samples)

        results = []

        with tru_app as recording:
            for idx, row in test_samples.iterrows():
                question = row['query']
                schema_info = row['schema_info']

                print(f"\n--- Sample {idx + 1}/{num_samples} ---")
                print(f"Question: {question}")

                # Generate and evaluate SQL
                result = self.generate_sql_with_evaluation(question, schema_info)
                results.append(result)
                print(f"Generated SQL: {result.generated_sql}")
                print(f"Ground Truth: {result.ground_truth_sql}")

                if result.error:
                    print(f"Error: {result.error}")

        return results, tru_app

    def display_evaluation_summary(self, tru_app: TruApp):
        """Display evaluation summary from TruLens"""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)

        # Get leaderboard
        leaderboard = self.session.get_leaderboard(app_ids=[tru_app.app_id])

        if not leaderboard.empty:
            print("\n📊 Performance Metrics:")
            print("-" * 40)

            for col in leaderboard.columns:
                if col not in ['app_id', 'app_name']:
                    value = leaderboard[col].iloc[0]
                    if isinstance(value, (int, float)):
                        print(f"{col}: {value:.3f}")
                    else:
                        print(f"{col}: {value}")

        # Get detailed records
        records_tuple = self.session.get_records_and_feedback(app_ids=[tru_app.app_id])
        records = records_tuple[0] # Access the DataFrame within the tuple
        print(f"Records : { records }")
        if not records.empty:
            print(f"\n📈 Processed {len(records)} queries")
            print(f"🎯 Average SQL Ground Truth Agreement: {records.get('SQL Ground Truth Agreement', pd.Series([0])).mean():.3f}")
            print(f"✅ Average SQL Correctness: {records.get('SQL Correctness', pd.Series([0])).mean():.3f}")
            print(f"🔗 Average Schema Relevance: {records.get('Schema Relevance', pd.Series([0])).mean():.3f}")

    def get_ground_truth_dataset_info(self):
        """Get information about the loaded ground truth dataset"""
        gt_df = self.session.get_ground_truth("duckdb_text2sql_groundtruth")

        print("📋 Ground Truth Dataset Info:")
        print("-" * 40)
        print(f"Total samples: {len(gt_df)}")
        print(f"Columns: {list(gt_df.columns)}")

        if len(gt_df) > 0:
            print(f"\nSample query: {gt_df.iloc[0]['query']}")
            print(f"Sample expected response: {gt_df.iloc[0]['expected_response']}")

        return gt_df

def main():
    """Main execution function"""

    TRULENS_DB_PATH = "my-trulens.sqlite3"
    TRULENS_DB_URL = f"sqlite:///{os.path.abspath(TRULENS_DB_PATH)}"
    print("🚀 Initializing DSPy + TruLens Text-to-SQL Ground Truth Evaluation")
    print("="*80)

    # Initialize the application
    # Create TruLens app for monitoring
    session = TruSession(database_url=TRULENS_DB_URL)
    app = DuckDBTextToSQLApp(session=session)

    # Display dataset info
    app.get_ground_truth_dataset_info()
    tru_app = TruApp(
        app=app,
        app_name="DSPy_Text2SQL_v1",
        feedbacks=app.feedback_functions,
        app_version="1.0.0",
    )

    # Run evaluation experiment
    results, tru_app = app.run_evaluation_experiment(num_samples=5,tru_app=tru_app)
    print("⏳ Waiting for feedback evaluations to complete...")
    time.sleep(10)  # Give enough time for async feedback
    
    # Display summary
    app.display_evaluation_summary(tru_app)
    
    # ADD THIS: Additional wait before cleanup
    print("🔄 Finalizing evaluation...")
    time.sleep(5)

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    main()
    