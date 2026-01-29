import os
import requests
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

base_url = "https://raw.githubusercontent.com/Running-Turtle1/MCM-ICM/master/"
# URL encoded path for "2025美赛特等奖"
folder_path_encoded = "2025%E7%BE%8E%E8%B5%9B%E7%89%B9%E7%AD%89%E5%A5%96"

target_root = r"C:\Users\EDWINJ\Desktop\浙大\竞赛\美赛\比赛期间\doc"

files = {
    "A": ["2500836.pdf", "2501567.pdf", "2501909.pdf", "2504218.pdf", "2511565.pdf"],
    "B": ["2501687.pdf", "2502617.pdf", "2503268.pdf", "2504448.pdf", "2505199.pdf", "2509557.pdf", "2517929.pdf"],
    "D": ["2504188.pdf", "2507692.pdf", "2516219.pdf", "2519935.pdf"]
}

def download_file(args):
    folder, filename = args
    # Construct URL: base + folder_path + / + subfolder + / + filename
    # Note: folder_path_encoded is already encoded
    url = f"{base_url}{folder_path_encoded}/{folder}/{filename}"
    
    target_dir = os.path.join(target_root, folder)
    target_path = os.path.join(target_dir, filename)
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    print(f"Downloading {folder}/{filename} from {url}...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            f.write(response.content)
        print(f"Saved to {target_path}")
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False

def main():
    tasks = []
    for folder, file_list in files.items():
        for filename in file_list:
            tasks.append((folder, filename))
            
    # Use threads to speed up
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(download_file, tasks))
        
    print(f"Download complete. Success: {sum(results)}/{len(results)}")

if __name__ == "__main__":
    main()
