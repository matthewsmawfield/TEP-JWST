#!/usr/bin/env python
"""
TEP-JWST: Main Pipeline Orchestrator
Runs the full analysis pipeline for the Chronological Shear paper.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Setup logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# PIPELINE STEPS
# ============================================================================
PIPELINE_STEPS = [
    {
        "name": "Step 1: Data Ingestion",
        "module": "steps.step_1_data_ingestion",
        "function": "run_data_ingestion",
        "kwargs": {"download": False, "process": True}
    },
    {
        "name": "Step 2: Mass-Age Analysis",
        "module": "steps.step_2_mass_age_analysis",
        "function": "run_mass_age_analysis",
        "kwargs": {"use_synthetic": True}  # Change to False for real data
    },
]

def run_step(step_config):
    """
    Run a single pipeline step.
    
    Parameters
    ----------
    step_config : dict
        Step configuration with module, function, and kwargs
    
    Returns
    -------
    success : bool
        True if step completed successfully
    """
    logger.info("=" * 60)
    logger.info(f"Running: {step_config['name']}")
    logger.info("=" * 60)
    
    try:
        # Import module
        module = __import__(step_config['module'], fromlist=[step_config['function']])
        func = getattr(module, step_config['function'])
        
        # Run function
        result = func(**step_config.get('kwargs', {}))
        
        logger.info(f"Completed: {step_config['name']}")
        return True, result
        
    except Exception as e:
        logger.error(f"Failed: {step_config['name']}")
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None

def run_pipeline(steps=None, stop_on_error=True):
    """
    Run the full analysis pipeline.
    
    Parameters
    ----------
    steps : list, optional
        List of step indices to run (default: all)
    stop_on_error : bool
        If True, stop pipeline on first error
    
    Returns
    -------
    results : dict
        Results from each step
    """
    logger.info("=" * 60)
    logger.info("TEP-JWST Analysis Pipeline")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    if steps is None:
        steps_to_run = PIPELINE_STEPS
    else:
        steps_to_run = [PIPELINE_STEPS[i] for i in steps if i < len(PIPELINE_STEPS)]
    
    results = {}
    success_count = 0
    fail_count = 0
    
    for step in steps_to_run:
        success, result = run_step(step)
        results[step['name']] = result
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            if stop_on_error:
                logger.error("Pipeline stopped due to error.")
                break
    
    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    logger.info(f"Completed: {success_count} steps")
    logger.info(f"Failed: {fail_count} steps")
    logger.info(f"Log file: {log_file}")
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TEP-JWST Analysis Pipeline")
    parser.add_argument("--steps", nargs="+", type=int, help="Step indices to run")
    parser.add_argument("--list", action="store_true", help="List available steps")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue pipeline even if a step fails")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Pipeline Steps:")
        print("-" * 40)
        for i, step in enumerate(PIPELINE_STEPS):
            print(f"  [{i}] {step['name']}")
        print()
        sys.exit(0)
    
    results = run_pipeline(
        steps=args.steps,
        stop_on_error=not args.continue_on_error
    )
