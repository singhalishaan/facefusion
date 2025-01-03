import os
import requests
import subprocess
import sys
from datetime import datetime
import cloudinary
import cloudinary.uploader

# Cloudinary configuration
cloudinary.config(
    cloud_name="dlnuvrqki",
    api_key="unused",
    api_secret="unused"
)

# Constants
SOURCE_URL = "https://res.cloudinary.com/dlnuvrqki/image/upload/v1735817569/marc_nf79yc.jpg"
TARGET_URL = "https://res.cloudinary.com/dlnuvrqki/image/upload/v1734590804/aiavatar/images/p0aysk01gffxc2ifj68v.jpg"

def print_status(message, status="INFO"):
    # Ensure immediate output flush for Docker logs
    timestamp = datetime.now().strftime("%H:%M:%S")
    message = f"[{timestamp}] [{status}] {message}"
    print(message, flush=True)

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print_status(f"Created directory: {path}")

def download_media(url, output_path):
    try:
        print_status(f"Downloading {output_path} from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print_status(f"Successfully downloaded {output_path}", "SUCCESS")
        return True
    except Exception as e:
        print_status(f"Error downloading {url}: {str(e)}", "ERROR")
        return False

def upload_to_cloudinary(file_path):
    try:
        print_status("Uploading result to Cloudinary...")
        response = cloudinary.uploader.upload(
            file_path,
            resource_type="image",
            upload_preset="rocyaab4",
            folder="facefusion_results"
        )
        print_status(f"Upload successful! URL: {response['secure_url']}", "SUCCESS")
        return response['secure_url']
    except Exception as e:
        print_status(f"Error uploading to Cloudinary: {str(e)}", "ERROR")
        return None

def stream_output(pipe, prefix):
    """Helper function to stream output from subprocess pipes"""
    for line in iter(pipe.readline, b''):
        if line:
            decoded_line = line.decode('utf-8').strip()
            if decoded_line:
                print_status(f"{prefix}: {decoded_line}")

def run_processing():
    try:
        print_status("Starting processing with predefined URLs")
        
        # Ensure required directories exist
        ensure_directory('.assets')
        ensure_directory('.caches')
        ensure_directory('.jobs')
        
        # Download files to .assets directory
        if not download_media(SOURCE_URL, ".assets/source.jpg"):
            return False
        if not download_media(TARGET_URL, ".assets/target.jpg"):
            return False
        
        print_status("Starting face fusion processing...")
        command = [
            "python", "facefusion.py", "run",
            "-s", ".assets/source.jpg",
            "-t", ".assets/target.jpg",
            "-o", ".assets/output.jpg",
            "--face-detector-model", "yoloface",
            "--face-swapper-model", "inswapper_128",
            "--face-swapper-pixel-boost", "512x512",
            "--execution-thread-count", "32",
            "--log-level", "info"
        ]
        
        # Run the process with real-time output streaming
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=False  # Changed to handle bytes
        )
        
        # Set up separate threads for stdout and stderr
        from threading import Thread
        stdout_thread = Thread(target=stream_output, args=(process.stdout, "STDOUT"))
        stderr_thread = Thread(target=stream_output, args=(process.stderr, "STDERR"))
        
        # Start threads
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Wait for output threads to complete
        stdout_thread.join()
        stderr_thread.join()
        
        if return_code == 0:
            print_status("Processing completed successfully!", "SUCCESS")
            output_path = ".assets/output.jpg"
            if os.path.exists(output_path):
                result_url = upload_to_cloudinary(output_path)
                if result_url:
                    print_status(f"Final result available at: {result_url}", "SUCCESS")
            return True
        else:
            print_status(f"Processing failed with return code: {return_code}", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"Error during processing: {str(e)}", "ERROR")
        return False
    finally:
        # Cleanup files
        files_to_clean = [
            ".assets/source.jpg",
            ".assets/target.jpg",
            ".assets/output.jpg"
        ]
        for file in files_to_clean:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print_status(f"Cleaned up {file}")
                except Exception as e:
                    print_status(f"Error cleaning up {file}: {str(e)}", "ERROR")

if __name__ == "__main__":
    success = run_processing()
    sys.exit(0 if success else 1)
