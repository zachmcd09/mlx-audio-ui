#!/usr/bin/env python3
"""
Master Orchestration Script for TTS Model Debugging

This script orchestrates the comprehensive debugging approach by running
all the specialized tools in sequence and generating a unified report.
"""
import os
import sys
import time
import argparse
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tts_debug.log')
    ]
)
logger = logging.getLogger('debug_tts_models')


class DebugOrchestrator:
    """
    Orchestrates the execution of all debugging tools and generates a comprehensive report.
    """
    
    def __init__(self, output_dir: str = "debug_report"):
        """
        Initialize the orchestrator.
        
        Args:
            output_dir: Directory to save the final report
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.start_time = time.time()
        self.results = {}
    
    def prepare_environment(self, venv_dir: str = "tts-debug-venv", cache_dir: str = "model_cache") -> bool:
        """
        Set up the isolated testing environment.
        
        Args:
            venv_dir: Path to create the virtual environment
            cache_dir: Path to create the model cache
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Setting up isolated testing environment...")
        
        try:
            cmd = [sys.executable, "setup_test_env.py", "--venv-dir", venv_dir, "--cache-dir", cache_dir]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info("Environment setup successful!")
            self.results["environment_setup"] = {
                "success": True,
                "venv_dir": venv_dir,
                "cache_dir": cache_dir,
                "stdout": result.stdout
            }
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set up environment: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            self.results["environment_setup"] = {
                "success": False,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
            return False
    
    def run_model_analysis(self) -> bool:
        """
        Run the model architecture analysis to understand parameter requirements.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Running model architecture analysis...")
        
        try:
            cmd = [sys.executable, "model_analyzer.py"]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info("Model analysis successful!")
            self.results["model_analysis"] = {
                "success": True,
                "stdout": result.stdout
            }
            
            # Try to parse analysis results
            analysis_dir = Path("model_analysis_results")
            if analysis_dir.exists():
                analysis_files = list(analysis_dir.glob("*.json"))
                analysis_results = {}
                
                for file in analysis_files:
                    try:
                        with open(file, 'r') as f:
                            analysis_results[file.stem] = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not parse analysis file {file}: {e}")
                
                self.results["model_analysis"]["results"] = analysis_results
            
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run model analysis: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            self.results["model_analysis"] = {
                "success": False,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
            return False
    
    def run_minimal_model_tests(self) -> bool:
        """
        Run the minimal model tests to isolate initialization issues.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Running minimal model tests...")
        
        try:
            cmd = [sys.executable, "minimal_model_factory.py"]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info("Minimal model tests successful!")
            
            # Extract test results from output
            success_rate_pattern = r"Minimal Model Success Rate: (\d+)/(\d+)"
            incremental_rate_pattern = r"Incremental Building Success Rate: (\d+)/(\d+)"
            
            import re
            minimal_match = re.search(success_rate_pattern, result.stdout)
            incremental_match = re.search(incremental_rate_pattern, result.stdout)
            
            success_counts = {}
            if minimal_match:
                success_counts["minimal_models"] = {
                    "success": int(minimal_match.group(1)),
                    "total": int(minimal_match.group(2))
                }
            
            if incremental_match:
                success_counts["incremental_building"] = {
                    "success": int(incremental_match.group(1)),
                    "total": int(incremental_match.group(2))
                }
            
            self.results["minimal_model_tests"] = {
                "success": True,
                "stdout": result.stdout,
                "success_counts": success_counts
            }
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run minimal model tests: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            self.results["minimal_model_tests"] = {
                "success": False,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
            return False
    
    def run_diagnostics(self) -> bool:
        """
        Run the detailed diagnostics to capture initialization issues.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Running model initialization diagnostics...")
        
        try:
            cmd = [sys.executable, "model_diagnostics.py"]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info("Diagnostics run successful!")
            
            # Find the generated report
            diagnostics_dir = Path("diagnostics_output")
            report_files = list(diagnostics_dir.glob("diagnostic_report_*.md"))
            report_path = None
            if report_files:
                report_path = str(sorted(report_files)[-1])  # Most recent report
            
            self.results["diagnostics"] = {
                "success": True,
                "stdout": result.stdout,
                "report_path": report_path
            }
            
            # Try to copy the report to our output directory
            if report_path:
                try:
                    import shutil
                    report_dest = self.output_dir / Path(report_path).name
                    shutil.copy2(report_path, report_dest)
                    logger.info(f"Copied diagnostic report to {report_dest}")
                except Exception as e:
                    logger.warning(f"Could not copy diagnostic report: {e}")
            
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run diagnostics: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            self.results["diagnostics"] = {
                "success": False,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
            return False
    
    def generate_unified_report(self) -> str:
        """
        Generate a unified HTML report from all test results.
        
        Returns:
            Path to the generated report
        """
        logger.info("Generating unified debug report...")
        
        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"tts_debug_report_{timestamp}.html"
        
        # Calculate duration
        duration = time.time() - self.start_time
        
        with open(report_path, 'w') as f:
            # HTML header
            f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TTS Model Debugging Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
        }
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            margin-top: 30px;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .success {
            color: #27ae60;
            font-weight: bold;
        }
        .failure {
            color: #e74c3c;
            font-weight: bold;
        }
        .details {
            background-color: #f1f1f1;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
            border-left: 4px solid #3498db;
        }
        pre {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .summary-box {
            display: inline-block;
            width: 200px;
            height: 100px;
            background-color: #f2f2f2;
            margin: 10px;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .summary-box h3 {
            margin-top: 0;
        }
        .summary-box .number {
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }
        .summary-box.success {
            background-color: #d5f5e3;
            color: #27ae60;
        }
        .summary-box.failure {
            background-color: #fadbd8;
            color: #e74c3c;
        }
    </style>
</head>
<body>
    <h1>TTS Model Debugging Report</h1>
    <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    <p>Total execution time: """ + f"{duration:.2f} seconds" + """</p>
            """)
            
            # Summary section
            f.write("""
    <h2>Summary</h2>
    <div class="container">
        <div class="summary-box """ + ("success" if self.results.get("environment_setup", {}).get("success", False) else "failure") + """">
            <h3>Environment</h3>
            <div class="number">""" + ("✓" if self.results.get("environment_setup", {}).get("success", False) else "✗") + """</div>
        </div>
        <div class="summary-box """ + ("success" if self.results.get("model_analysis", {}).get("success", False) else "failure") + """">
            <h3>Analysis</h3>
            <div class="number">""" + ("✓" if self.results.get("model_analysis", {}).get("success", False) else "✗") + """</div>
        </div>
        <div class="summary-box """ + ("success" if self.results.get("minimal_model_tests", {}).get("success", False) else "failure") + """">
            <h3>Models</h3>
            <div class="number">""" + ("✓" if self.results.get("minimal_model_tests", {}).get("success", False) else "✗") + """</div>
        </div>
        <div class="summary-box """ + ("success" if self.results.get("diagnostics", {}).get("success", False) else "failure") + """">
            <h3>Diagnostics</h3>
            <div class="number">""" + ("✓" if self.results.get("diagnostics", {}).get("success", False) else "✗") + """</div>
        </div>
    </div>
            """)
            
            # Environment setup section
            f.write("""
    <h2>Environment Setup</h2>
    <div class="container">
            """)
            
            env_setup = self.results.get("environment_setup", {})
            if env_setup.get("success", False):
                f.write(f"""
        <p class="success">Environment setup completed successfully!</p>
        <p>Virtual environment: {env_setup.get('venv_dir', 'N/A')}</p>
        <p>Cache directory: {env_setup.get('cache_dir', 'N/A')}</p>
                """)
            else:
                f.write(f"""
        <p class="failure">Environment setup failed!</p>
        <p>Error: {env_setup.get('error', 'Unknown error')}</p>
        <div class="details">
            <h4>Error Output:</h4>
            <pre>{env_setup.get('stderr', 'No error output available')}</pre>
        </div>
                """)
            
            f.write("""
    </div>
            """)
            
            # Model Analysis section
            f.write("""
    <h2>Model Architecture Analysis</h2>
    <div class="container">
            """)
            
            model_analysis = self.results.get("model_analysis", {})
            if model_analysis.get("success", False):
                f.write("""
        <p class="success">Model architecture analysis completed successfully!</p>
                """)
                
                # If we have analysis results, show a summary
                analysis_results = model_analysis.get("results", {})
                if analysis_results:
                    f.write("""
        <h3>Analysis Results Summary</h3>
        <table>
            <tr>
                <th>Model</th>
                <th>Parameters</th>
                <th>Module</th>
            </tr>
                    """)
                    
                    for model_name, result in analysis_results.items():
                        f.write(f"""
            <tr>
                <td>{result.get('model_class', 'Unknown')}</td>
                <td>{len(result.get('parameters', []))}</td>
                <td>{result.get('module', 'Unknown')}</td>
            </tr>
                        """)
                    
                    f.write("""
        </table>
                    """)
            else:
                f.write(f"""
        <p class="failure">Model architecture analysis failed!</p>
        <p>Error: {model_analysis.get('error', 'Unknown error')}</p>
        <div class="details">
            <h4>Error Output:</h4>
            <pre>{model_analysis.get('stderr', 'No error output available')}</pre>
        </div>
                """)
            
            f.write("""
    </div>
            """)
            
            # Minimal Model Tests section
            f.write("""
    <h2>Minimal Model Tests</h2>
    <div class="container">
            """)
            
            minimal_tests = self.results.get("minimal_model_tests", {})
            if minimal_tests.get("success", False):
                f.write("""
        <p class="success">Minimal model tests completed successfully!</p>
                """)
                
                # Show success counts
                success_counts = minimal_tests.get("success_counts", {})
                if success_counts:
                    f.write("""
        <h3>Test Results</h3>
        <table>
            <tr>
                <th>Test Type</th>
                <th>Success</th>
                <th>Total</th>
                <th>Success Rate</th>
            </tr>
                    """)
                    
                    for test_type, counts in success_counts.items():
                        success = counts.get("success", 0)
                        total = counts.get("total", 0)
                        rate = (success / total * 100) if total > 0 else 0
                        
                        f.write(f"""
            <tr>
                <td>{test_type.replace('_', ' ').title()}</td>
                <td>{success}</td>
                <td>{total}</td>
                <td>{rate:.1f}%</td>
            </tr>
                        """)
                    
                    f.write("""
        </table>
                    """)
            else:
                f.write(f"""
        <p class="failure">Minimal model tests failed!</p>
        <p>Error: {minimal_tests.get('error', 'Unknown error')}</p>
        <div class="details">
            <h4>Error Output:</h4>
            <pre>{minimal_tests.get('stderr', 'No error output available')}</pre>
        </div>
                """)
            
            f.write("""
    </div>
            """)
            
            # Diagnostics section
            f.write("""
    <h2>Model Initialization Diagnostics</h2>
    <div class="container">
            """)
            
            diagnostics = self.results.get("diagnostics", {})
            if diagnostics.get("success", False):
                f.write("""
        <p class="success">Diagnostic testing completed successfully!</p>
                """)
                
                # Show report link if available
                report_path = diagnostics.get("report_path")
                if report_path:
                    report_name = Path(report_path).name
                    f.write(f"""
        <p>Detailed diagnostic report: <a href="{report_name}">{report_name}</a></p>
                    """)
            else:
                f.write(f"""
        <p class="failure">Diagnostic testing failed!</p>
        <p>Error: {diagnostics.get('error', 'Unknown error')}</p>
        <div class="details">
            <h4>Error Output:</h4>
            <pre>{diagnostics.get('stderr', 'No error output available')}</pre>
        </div>
                """)
            
            f.write("""
    </div>
            """)
            
            # Recommendations section
            f.write("""
    <h2>Recommendations</h2>
    <div class="container">
        <h3>Based on Analysis Results</h3>
        <ul>
            """)
            
            # Generate recommendations based on test results
            if not all(test.get("success", False) for test in self.results.values()):
                f.write("""
            <li>Re-run the tests with more detailed logging enabled.</li>
                """)
            
            minimal_success_counts = self.results.get("minimal_model_tests", {}).get("success_counts", {})
            minimal_models = minimal_success_counts.get("minimal_models", {})
            if minimal_models.get("success", 0) < minimal_models.get("total", 0):
                f.write("""
            <li>Focus on the minimal model failures first, as these represent core initialization issues.</li>
                """)
            
            incremental_building = minimal_success_counts.get("incremental_building", {})
            if incremental_building.get("success", 0) < incremental_building.get("total", 0):
                f.write("""
            <li>Examine the incremental building failures to identify which components are causing issues.</li>
                """)
            
            f.write("""
            <li>Follow the parameter validation recommendations in the diagnostic report.</li>
            <li>Implement a progressive loading pipeline with fallbacks for problematic components.</li>
            <li>Consider adding a parameter adaptation layer to handle differences between model versions.</li>
            <li>Cache successful model configurations locally to reduce network dependencies.</li>
        </ul>
        
        <h3>Implementation Strategy</h3>
        <ol>
            <li>Fix core parameter validation to catch invalid inputs early</li>
            <li>Implement a robust parameter adaptation layer for Encodec</li>
            <li>Add graceful fallbacks when specific components fail</li>
            <li>Create a model registry to track known-good configurations</li>
            <li>Implement lazy loading to avoid startup failures</li>
        </ol>
    </div>
            """)
            
            # HTML footer
            f.write("""
</body>
</html>
            """)
        
        logger.info(f"Unified report generated at: {report_path}")
        return str(report_path)
    
    def run_all_tests(self, skip_environment: bool = False) -> bool:
        """
        Run all debug tests in sequence.
        
        Args:
            skip_environment: Whether to skip environment setup
            
        Returns:
            True if all tests run (regardless of their result), False if critical error
        """
        logger.info("Starting comprehensive TTS model debugging...")
        
        # Setup environment
        if not skip_environment:
            if not self.prepare_environment():
                logger.warning("Environment setup failed! Proceeding with existing environment.")
        
        # Run all tests
        self.run_model_analysis()
        self.run_minimal_model_tests()
        self.run_diagnostics()
        
        # Generate report
        report_path = self.generate_unified_report()
        
        logger.info(f"All debug tests completed. Report available at: {report_path}")
        return True


def main():
    """Run the master debug script."""
    parser = argparse.ArgumentParser(description="Orchestrate TTS model debugging")
    parser.add_argument("--skip-environment", action="store_true", 
                      help="Skip environment setup and use existing environment")
    parser.add_argument("--output-dir", default="debug_report",
                      help="Directory to save debug reports")
    args = parser.parse_args()
    
    print("TTS Model Debugging Orchestrator")
    print("================================")
    print(f"Starting comprehensive debugging at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    orchestrator = DebugOrchestrator(output_dir=args.output_dir)
    success = orchestrator.run_all_tests(skip_environment=args.skip_environment)
    
    if success:
        print("\n✅ Debugging process completed.")
        print(f"Report available in the {args.output_dir} directory.")
        return 0
    else:
        print("\n❌ Debugging process failed due to critical error.")
        print("Check the tts_debug.log file for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
