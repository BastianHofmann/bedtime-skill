# for testing move to ml dir
import requests
from pathlib import Path

def upload():
    path = Path(__file__).parent / "weights" / "TX3"
    request = [('model_id', (None, "SS2"))]
    url = "http://127.0.0.1:8000/uploadmodel"

    for extension in ["*.json", "*.h5"]:
        for current_file in path.glob(extension):
            request.append(('files', (current_file.name, open(current_file, 'rb'))))
    r = requests.post(url, files=request)
    if r.status_code != 200:
        print("The Model data could not been uploaded")
        return False
    else:
        return True

if __name__ == "__main__":
    upload()