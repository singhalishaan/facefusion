import os
import requests
import subprocess
import sys
from datetime import datetime

# Constants
CLOUDINARY_URL = "https://api.cloudinary.com/v1_1/dlnuvrqki/image/upload"
UPLOAD_PRESET = "rocyaab4"
SOURCE_URL = "https://res.cloudinary.com/dlnuvrqki/image/upload/v1735817569/marc_nf79yc.jpg"
TARGET_URL = "https://res.cloudinary.com/dlnuvrqki/image/upload/v1734590804/aiavatar/images/p0aysk01gffxc2ifj68v.jpg"

def print_status(message, status="INFO"):
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

def upload_to_cloudinary_unsigned(file_path):
    try:
        print_status("Uploading result to Cloudinary (unsigned)...")
        with open(file_path, "rb") as file:
            response = requests.post(
                CLOUDINARY_URL,
                files={"file": file},
                data={"upload_preset": UPLOAD_PRESET}
            )
        response.raise_for_status()
        upload_response = response.json()
        secure_url = upload_response.get("secure_url")
        if secure_url:
            print_status(f"Upload successful! URL: {secure_url}", "SUCCESS")
            return secure_url
        else:
            print_status(f"Unexpected response: {upload_response}", "ERROR")
            return None
    except Exception as e:
        print_status(f"Error uploading to Cloudinary: {str(e)}", "ERROR")
        return None

def run_processing():
    try:
        print_status("Starting processing with predefined URLs")
        ensure_directory('.assets')
        ensure_directory('.caches')
        ensure_directory('.jobs')
        if not download_media(SOURCE_URL, ".assets/source.jpg"):
            return False
        if not download_media(TARGET_URL, ".assets/target.jpg"):
            return False
        print_status("Starting face fusion processing...")
        command = [
            "python", "facefusion.py", "headless-run",
            "-s", ".assets/source.jpg",
            "-t", ".assets/target.jpg",
            "-o", ".assets/output.jpg",
            "--face-detector-model", "yoloface",
            "--face-swapper-model", "inswapper_128",
            "--face-swapper-pixel-boost", "512x512",
            "--execution-thread-count", "32",
            "--log-level", "debug"
        ]
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode == 0:
            print_status("Processing completed successfully!", "SUCCESS")
            output_path = ".assets/output.jpg"
            if os.path.exists(output_path):
                result_url = upload_to_cloudinary_unsigned(output_path)
                if result_url:
                    print_status(f"Final result available at: {result_url}", "SUCCESS")
            return True
        else:
            print_status(f"Processing failed with return code: {process.returncode}", "ERROR")
            print_status(process.stderr, "STDERR")
            return False
    except Exception as e:
        print_status(f"Error during processing: {str(e)}", "ERROR")
        return False
    finally:
        files_to_clean = [".assets/source.jpg", ".assets/target.jpg", ".assets/output.jpg"]
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
