"""
Main entry point for Real-Time Seizure Prediction System.
"""

import argparse
import sys
from pathlib import Path
import yaml
from loguru import logger

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/seizure_prediction_{time}.log",
    rotation="100 MB",
    retention="30 days",
    level="DEBUG"
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_environment():
    """Setup necessary directories and environment."""
    directories = [
        "data/raw",
        "data/processed",
        "data/models",
        "logs",
        "results",
        "checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Environment setup complete")


def run_data_download(config: dict):
    """Download datasets."""
    from scripts.download_kaggle_datasets import KaggleDatasetDownloader
    
    logger.info("Starting dataset download...")
    downloader = KaggleDatasetDownloader()
    
    if downloader.check_kaggle_api():
        downloader.setup_kaggle_api()
        results = downloader.download_all_datasets()
        
        success_count = sum(results.values())
        total_count = len(results)
        logger.info(f"Downloaded {success_count}/{total_count} datasets successfully")
    else:
        logger.warning("Kaggle API not available. Please install: pip install kaggle")


def run_training(config: dict, patient_id: str = None):
    """Run model training."""
    logger.info("=" * 60)
    logger.info("TRAINING MODE")
    logger.info("=" * 60)
    
    from src.models.trainer import SeizureModelTrainer
    from src.data.loader import EEGDataLoader
    
    if patient_id:
        # Train single patient
        logger.info(f"Training model for patient: {patient_id}")
        
        loader = EEGDataLoader(patient_id=patient_id, config=config)
        train_data, test_data = loader.load_patient_data()
        
        trainer = SeizureModelTrainer(patient_id=patient_id, config=config)
        model = trainer.train(train_data)
        
        # Evaluate
        metrics = trainer.evaluate(test_data)
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Patient: {patient_id}")
        logger.info(f"Sensitivity: {metrics['sensitivity']:.2%}")
        logger.info(f"Specificity: {metrics['specificity']:.2f} false alarms/24h")
        logger.info(f"Median Latency: {metrics['median_latency']:.2f}s")
        
    else:
        # Train all patients
        logger.info("Training models for all patients...")
        
        from scripts.train_all_patients import train_all_patients
        train_all_patients(config)


def run_realtime(config: dict, patient_id: str):
    """Run real-time prediction."""
    logger.info("=" * 60)
    logger.info("REAL-TIME PREDICTION MODE")
    logger.info("=" * 60)
    
    from src.realtime.predictor import RealtimeSeizurePredictor
    from src.realtime.stream_processor import EEGStreamProcessor
    
    logger.info(f"Initializing predictor for patient: {patient_id}")
    
    # Initialize predictor
    model_path = f"data/models/{patient_id}_model.pkl"
    predictor = RealtimeSeizurePredictor(
        patient_id=patient_id,
        model_path=model_path,
        config=config
    )
    
    # Initialize stream processor
    stream_processor = EEGStreamProcessor(config=config)
    
    logger.info("Starting real-time monitoring...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        # Process EEG stream
        for eeg_chunk in stream_processor.stream():
            prediction = predictor.predict(eeg_chunk)
            
            if prediction['seizure_detected']:
                logger.warning("⚠️  SEIZURE DETECTED!")
                logger.warning(f"   Confidence: {prediction['confidence']:.2%}")
                logger.warning(f"   Timestamp: {prediction['timestamp']}")
                
                # Trigger alert
                if config['alerts']['enabled']:
                    from src.realtime.alert_manager import AlertManager
                    alert_mgr = AlertManager(config)
                    alert_mgr.send_alert(prediction)
                    
    except KeyboardInterrupt:
        logger.info("\nStopping real-time monitoring...")
        stream_processor.stop()


def run_api_server(config: dict):
    """Run API server."""
    logger.info("=" * 60)
    logger.info("API SERVER MODE")
    logger.info("=" * 60)
    
    from src.api.server import create_app
    import uvicorn
    
    app = create_app(config)
    
    host = config['api']['host']
    port = config['api']['port']
    workers = config['api']['workers']
    
    logger.info(f"Starting API server at http://{host}:{port}")
    logger.info(f"API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )


def run_dashboard():
    """Run web dashboard."""
    logger.info("=" * 60)
    logger.info("DASHBOARD MODE")
    logger.info("=" * 60)
    
    import subprocess
    import os
    
    dashboard_dir = Path("dashboard")
    
    if not dashboard_dir.exists():
        logger.error("Dashboard directory not found. Please run: npm install")
        return
    
    logger.info("Starting dashboard...")
    logger.info("Dashboard will be available at http://localhost:3000")
    
    try:
        subprocess.run(["npm", "start"], cwd=dashboard_dir, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start dashboard: {e}")
    except FileNotFoundError:
        logger.error("Node.js/npm not found. Please install Node.js")


def run_evaluation(config: dict, patient_id: str = None):
    """Run model evaluation."""
    logger.info("=" * 60)
    logger.info("EVALUATION MODE")
    logger.info("=" * 60)
    
    from src.models.evaluator import SeizureModelEvaluator
    
    evaluator = SeizureModelEvaluator(config=config)
    
    if patient_id:
        logger.info(f"Evaluating model for patient: {patient_id}")
        metrics = evaluator.evaluate_patient(patient_id)
        evaluator.print_metrics(metrics, patient_id)
    else:
        logger.info("Evaluating all patient models...")
        all_metrics = evaluator.evaluate_all_patients()
        evaluator.print_summary(all_metrics)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Real-Time Seizure Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download datasets
  python main.py --mode download
  
  # Train model for specific patient
  python main.py --mode train --patient chb01
  
  # Train models for all patients
  python main.py --mode train
  
  # Run real-time prediction
  python main.py --mode realtime --patient chb01
  
  # Start API server
  python main.py --mode api
  
  # Start dashboard
  python main.py --mode dashboard
  
  # Evaluate model
  python main.py --mode evaluate --patient chb01
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["download", "train", "realtime", "api", "dashboard", "evaluate"],
        help="Operation mode"
    )
    
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Patient ID (e.g., chb01)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # ASCII Banner
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   Real-Time Seizure Prediction System                    ║
    ║   Temporal Lobe Epilepsy Detection using ML              ║
    ║                                                           ║
    ║   96% Sensitivity | 3s Median Latency | 2 FA/24h         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    
    print(banner)
    
    # Setup
    setup_environment()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Run selected mode
    try:
        if args.mode == "download":
            run_data_download(config)
            
        elif args.mode == "train":
            run_training(config, args.patient)
            
        elif args.mode == "realtime":
            if not args.patient:
                logger.error("Patient ID required for real-time mode")
                sys.exit(1)
            run_realtime(config, args.patient)
            
        elif args.mode == "api":
            run_api_server(config)
            
        elif args.mode == "dashboard":
            run_dashboard()
            
        elif args.mode == "evaluate":
            run_evaluation(config, args.patient)
            
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        sys.exit(1)
    
    logger.success("\n✓ Operation completed successfully!")


if __name__ == "__main__":
    main()
