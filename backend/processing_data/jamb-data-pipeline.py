import os
import sys
from pathlib import Path
import logging

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from mongo_setup import setup_jamb_db
from data_collection import JambDataCollector
from jamb_extractor import JambQuestionExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JambDataPipeline:
    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
        self.database_name = os.getenv("MONGO_DB_NAME", "jamb_db")
        
        logger.info(f"Initializing pipeline with MongoDB URI: {self.mongo_uri}")
        logger.info(f"Database name: {self.database_name}")
        
        self.db = None
        self.collector = None
        self.extractor = None
        
    
        self.base_data_dir = Path(__file__).parent / "processing_data"
        logger.info(f"Base data directory: {self.base_data_dir}")
        
    def setup_database(self):
        """Setup database connection and collections with retry logic"""
        logger.info("Setting up database connection...")
        
        try:
            # Use the setup function from mongo_setup.py which has retry logic
            self.db = setup_jamb_db()
            
            if self.db is None:
                logger.error("Failed to setup database connection")
                return False
            
            # Test the connection
            db_info = self.db.test_connection()
            if db_info:
                logger.info("Database connection verified successfully")
                logger.info(f"Connected to database: {db_info['database_name']}")
                logger.info(f"Database size: {db_info['database_size_bytes']} bytes")
                
                # Log collection info
                for collection_name, info in db_info['collections'].items():
                    logger.info(f"Collection '{collection_name}': {info['document_count']} documents")
                
                return True
            else:
                logger.error("Database connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"Database setup failed with exception: {e}")
            return False
        
    def setup_collector(self):
        """Setup the data collector"""
        try:
            logger.info("Setting up data collector...")
            
            # Ensure base directory exists
            self.base_data_dir.mkdir(exist_ok=True)
            
            # Check if PDF directory exists and list contents
            pdf_dir = self.base_data_dir / "pdf_documents"
            if pdf_dir.exists():
                pdf_files = list(pdf_dir.glob("**/*.pdf"))
                logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
                for pdf_file in pdf_files:
                    logger.info(f"  - {pdf_file.name}")
            else:
                logger.warning(f"PDF directory does not exist: {pdf_dir}")
                pdf_dir.mkdir(exist_ok=True)
                logger.info(f"Created PDF directory: {pdf_dir}")
            
            self.collector = JambDataCollector(
                db_instance=self.db,
                base_data_dir=str(self.base_data_dir)
            )
            
            logger.info("Data collector setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Data collector setup failed: {e}")
            return False
    
    def setup_extractor(self):
        """Setup the question extractor"""
        try:
            logger.info("Setting up question extractor...")
            self.extractor = JambQuestionExtractor()
            logger.info("Question extractor setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Question extractor setup failed: {e}")
            return False
    
    def collect_web_data(self):
        """Collect data from web sources"""
        try:
            logger.info("Starting web data collection...")
            collected_count = self.collector.collect_web_data()
            
            if collected_count > 0:
                logger.info(f"Web data collection complete: {collected_count} documents processed")
            else:
                logger.warning("No web data collected")
            
            return collected_count
            
        except Exception as e:
            logger.error(f"Web data collection failed: {e}")
            return 0
    
    def collect_pdf_data(self):
        """Collect data from PDF sources"""
        try:
            logger.info("Starting PDF data collection...")
            
            # Double-check PDF directory before collection
            pdf_dir = self.base_data_dir / "pdf_documents"
            if not pdf_dir.exists():
                logger.error(f"PDF directory does not exist: {pdf_dir}")
                return 0
            
            pdf_files = list(pdf_dir.glob("**/*.pdf"))
            logger.info(f"About to process {len(pdf_files)} PDF files")
            
            if not pdf_files:
                logger.warning(f"No PDF files found in {pdf_dir}")
                logger.info("Please place JAMB PDF files in the processing_data/pdf_documents/ directory")
                return 0
            
            collected_count = self.collector.collect_pdf_data()
            
            if collected_count > 0:
                logger.info(f"PDF data collection complete: {collected_count} documents processed")
            else:
                logger.warning("No PDF data collected")
            
            return collected_count
            
        except Exception as e:
            logger.error(f"PDF data collection failed: {e}")
            return 0
    
    def organize_data(self):
        """Organize collected data by year"""
        try:
            logger.info("Starting data organization...")
            self.collector.organize_by_year()
            logger.info("Data organization complete")
            return True
            
        except Exception as e:
            logger.error(f"Data organization failed: {e}")
            return False
    
    def get_pipeline_statistics(self):
        """Get statistics about the collected data"""
        try:
            if not self.db:
                return None
                
            stats = {}
            
            raw_docs_collection = self.db.get_collection('raw_documents')
            if raw_docs_collection:
                stats['raw_documents'] = raw_docs_collection.count_documents({})
                
                web_count = raw_docs_collection.count_documents({'type': 'web'})
                pdf_count = raw_docs_collection.count_documents({'type': 'pdf'})
                stats['web_documents'] = web_count
                stats['pdf_documents'] = pdf_count
            
            questions_collection = self.db.get_collection('processed_questions')
            if questions_collection:
                stats['total_questions'] = questions_collection.count_documents({})
                
                subjects_pipeline = [
                    {"$group": {"_id": "$subject", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}}
                ]
                subject_counts = list(questions_collection.aggregate(subjects_pipeline))
                stats['questions_by_subject'] = {item['_id']: item['count'] for item in subject_counts}
                
                years_pipeline = [
                    {"$group": {"_id": "$year", "count": {"$sum": 1}}},
                    {"$sort": {"_id": -1}}
                ]
                year_counts = list(questions_collection.aggregate(years_pipeline))
                stats['questions_by_year'] = {item['_id']: item['count'] for item in year_counts}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting pipeline statistics: {e}")
            return None
    
    def print_statistics(self):
        """Print pipeline statistics"""
        stats = self.get_pipeline_statistics()
        if stats:
            print("\n" + "="*50)
            print("JAMB DATA PIPELINE STATISTICS")
            print("="*50)
            print(f"Total Raw Documents: {stats.get('raw_documents', 0)}")
            print(f"  - Web Documents: {stats.get('web_documents', 0)}")
            print(f"  - PDF Documents: {stats.get('pdf_documents', 0)}")
            print(f"Total Questions Extracted: {stats.get('total_questions', 0)}")
            
            if stats.get('questions_by_subject'):
                print("\nQuestions by Subject:")
                for subject, count in stats['questions_by_subject'].items():
                    print(f"  - {subject}: {count}")
            
            if stats.get('questions_by_year'):
                print("\nQuestions by Year:")
                for year, count in stats['questions_by_year'].items():
                    year_str = str(year) if year else "Unknown"
                    print(f"  - {year_str}: {count}")
            
            print("="*50)
        else:
            print("Could not retrieve pipeline statistics")
        
    def run_pipeline(self):
        """Run the complete data collection pipeline"""
        print("Starting JAMB Data Collection Pipeline...")
        
        try:
            if not self.setup_database():
                return False
                        
            if not self.setup_collector():
                return False
                        
            if not self.setup_extractor():
                return False
                        
            self.collect_web_data()
                        
            self.collect_pdf_data()
                        
            self.organize_data()
            
                        
            print("\nPipeline completed successfully")
            return True
                    
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            print(f"Pipeline failed: {e}")
            return False
                
        finally:
            if self.db:
                self.db.close_connection()

def main():
    pipeline = JambDataPipeline()
    
    if pipeline.run_pipeline():
        print("JAMB Data Collection Pipeline completed successfully!")
    else:
        print("JAMB Data Collection Pipeline failed!")

if __name__ == "__main__":
    main()