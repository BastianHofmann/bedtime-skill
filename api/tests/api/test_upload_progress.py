# for testing move to ml dir
import requests
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from pathlib import Path
from tqdm import tqdm


class TqdmUpTo(tqdm):
    # Source https://github.com/tqdm/tqdm/issues/478
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n) # will also set self.n = b * bsize 


def create_callback(total_size):
    
    pbar = TqdmUpTo(total=total_size, unit="B", unit_scale=True, desc="Upload files",initial=0)
    def callback(monitor):
        pbar.update_to(monitor.bytes_read)

    return callback


def upload():
    # source https://stackoverflow.com/questions/13909900/progress-of-python-requests-post
    path = Path(__file__).parent / "weights" / "TX3"
    request = [('model_id', (None, "SS2"))]
    url = "http://127.0.0.1:8000/uploadmodel"

    for extension in ["*.json", "*.h5"]:
        for current_file in path.glob(extension):
            request.append(('files', (current_file.name, open(current_file, 'rb'))))
    encoder = MultipartEncoder(request)
    callback = create_callback(encoder.len)
    m = MultipartEncoderMonitor(encoder, callback)

    r = requests.post(url, data=m, headers={'Content-Type': m.content_type})
    if r.status_code != 200:
        print("The Model data could not been uploaded")
        return False
    else:
        return True

if __name__ == "__main__":
    upload()