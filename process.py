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
    api_key="unused",  # Not needed for unsigned uploads
    api_secret="unused"  # Not needed for unsigned uploads
)

# Constants
SOURCE_URL = "https://res.cloudinary.com/dlnuvrqki/image/upload/v1735817569/marc_nf79yc.jpg"
TARGET_URL = "https://res.cloudinary.com/dlnuvrqki/video/upload/v1735817526/target_jvspkb.mp4"

def print_status(message, status="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    message = f"[{timestamp}] [{status}] {message}"
    print(message)
    sys.stdout.flush()

def download_media(url, output_path):
    try:
        print_status(f"Downloading {output_path} from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
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
            resource_type="video",
            upload_preset="rocyaab4",
            folder="facefusion_results"
        )
        print_status(f"Upload successful! URL: {response['secure_url']}", "SUCCESS")
        return response['secure_url']
    except Exception as e:
        print_status(f"Error uploading to Cloudinary: {str(e)}", "ERROR")
        return None

def run_processing():
    try:
        print_status("Starting processing with predefined URLs")
        
        if not download_media(SOURCE_URL, "source.jpg"):
            return False
        if not download_media(TARGET_URL, "target.mp4"):
            return False
        
        print_status("Starting face fusion processing...")
        command = [
            "python", "facefusion.py", "run",
            "-s", "source.jpg",
            "-t", "target.mp4",
            "-o", "output.mp4",
            "--face-detector-model", "yoloface",
            "--face-swapper-model", "inswapper_128",
            "--face-swapper-pixel-boost", "512x512",
            "--execution-providers", "cuda",
            "--execution-thread-count", "32",
            "--log-level", "info"
        ]
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()
            
            if output:
                print_status(output.strip())
            if error:
                print_status(error.strip(), "ERROR")
            
            if output == "" and error == "" and process.poll() is not None:
                break
        
        return_code = process.poll()
        
        if return_code == 0:
            print_status("Processing completed successfully!", "SUCCESS")
            # Upload the result to Cloudinary
            if os.path.exists("output.mp4"):
                result_url = upload_to_cloudinary("output.mp4")
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
        for file in ["source.jpg", "target.mp4", "output.mp4"]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print_status(f"Cleaned up {file}")
                except Exception as e:
                    print_status(f"Error cleaning up {file}: {str(e)}", "ERROR")

if __name__ == "__main__":
    success = run_processing()
    sys.exit(0 if success else 1)
