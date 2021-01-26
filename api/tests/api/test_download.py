# for testing move to ml dir
from tqdm import tqdm
from pathlib import Path
import requests

def download():
    # source https://towardsdatascience.com/how-to-download-files-using-python-part-2-19b95be4cdb5
    path = Path(__file__).parent / "weights" / "TZ3"
    path.mkdir(parents=True, exist_ok=True)
    filename = "config.json"
    file = path / filename

    url = "http://127.0.0.1:8000/downloadmodelfile"
    request = {"model_id": "SS2", "filename": filename}
    
    r = requests.post(url, stream=True, allow_redirects=True, json=request)
    total_size = int(r.headers.get('content-length'))
    initial_pos = 0
    with open(file,'wb') as f: 
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename,initial=initial_pos) as pbar:
            for ch in r.iter_content(chunk_size=1024):                             
                if ch:
                    f.write(ch) 
                    pbar.update(len(ch))

if __name__ == "__main__":
    download()