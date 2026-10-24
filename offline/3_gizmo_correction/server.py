import http.server
import socketserver
import os
import json
import urllib.parse
import shutil
import re

PORT = 8000
# The server is run from ~/Research/website/ so sign_annotation is at ../sign_annotation/
# For local testing in the skinning/src folder, we can create a dummy sign_annotation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGN_ANNOTATION_DIR = os.path.join(os.path.dirname(BASE_DIR), 'sign_annotation')


class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


# Create dummy directory for local testing if it doesn't exist
if not os.path.exists(SIGN_ANNOTATION_DIR):
    os.makedirs(SIGN_ANNOTATION_DIR)
    # create some dummy data
    for d in ['apple', 'school', 'car']:
        d_path = os.path.join(SIGN_ANNOTATION_DIR, d)
        os.makedirs(os.path.join(d_path, 'smplx_keyframes'), exist_ok=True)
        os.makedirs(os.path.join(d_path, 'ISL_keyframes'), exist_ok=True)


class APIHandler(http.server.SimpleHTTPRequestHandler):
    # NOTE: no end_headers() override anymore.
    # Previously this class forced Cache-Control: no-store on EVERY response
    # (images, JS, GLB models, everything), which meant the browser could
    # never cache anything and re-fetched/re-decoded assets on every load.
    # Now caching headers are applied selectively, only where staleness
    # actually matters (API JSON + NPZ files that can be edited/saved).

    def _no_cache(self):
        """Call this before end_headers() for responses that must never be cached."""
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)

        # 1. API: Get all glosses
        if parsed_path.path == '/api/glosses':
            query = urllib.parse.parse_qs(parsed_path.query)
            category = query.get('category', ['raw'])[0]

            target_dir = os.path.join(SIGN_ANNOTATION_DIR, category)
            folders = []

            if os.path.exists(target_dir):
                for f in os.listdir(target_dir):
                    if os.path.isdir(os.path.join(target_dir, f)):
                        has_corrected = False
                        if category == 'raw':
                            corrected_path = os.path.join(SIGN_ANNOTATION_DIR, 'corrected', f"{f}_corrected")
                            has_corrected = os.path.exists(corrected_path)
                        folders.append({"name": f, "has_corrected": has_corrected})

            folders.sort(key=lambda x: x["name"])

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self._no_cache()
            self.end_headers()
            self.wfile.write(json.dumps({"glosses": folders}).encode())
            return

        # 2. API: Get files for a gloss
        if parsed_path.path == '/api/gloss_files':
            query = urllib.parse.parse_qs(parsed_path.query)
            folder = query.get('folder', [''])[0]
            category = query.get('category', ['raw'])[0]
            if not folder:
                self.send_error(400, "Missing folder parameter")
                return

            npz_dir = os.path.join(SIGN_ANNOTATION_DIR, category, folder)

            # Extract base name for pics directory (e.g. 'school_corrected' -> 'school')
            base_name = folder.replace('_corrected', '') if category == 'corrected' else folder
            img_dir = os.path.join(SIGN_ANNOTATION_DIR, 'pics', base_name)

            frames = []
            if os.path.exists(npz_dir):
                files = os.listdir(npz_dir)
                for f in files:
                    if f.endswith('.npz'):
                        match = re.search(r'\d+', f)
                        if match:
                            frame_idx = int(match.group(0))

                            # Build URLs that the browser can fetch
                            npz_url = f"/sign_annotation/{category}/{folder}/{f}"

                            # Search for matching image in pics
                            img_url = None
                            if os.path.exists(img_dir):
                                for img_f in os.listdir(img_dir):
                                    if img_f.lower().endswith(('.jpg', '.png', '.jpeg')):
                                        img_match = re.search(r'\d+', img_f)
                                        if img_match and int(img_match.group(0)) == frame_idx:
                                            img_url = f"/sign_annotation/pics/{base_name}/{img_f}"
                                            break

                            frames.append({
                                "index": frame_idx,
                                "npz_url": npz_url,
                                "img_url": img_url,
                                "filename": f
                            })

            # Sort by frame index
            frames.sort(key=lambda x: x['index'])

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self._no_cache()
            self.end_headers()
            self.wfile.write(json.dumps({"frames": frames}).encode())
            return

        # 3. Serve static files from sign_annotation
        if parsed_path.path.startswith('/sign_annotation/'):
            # Remove the leading /sign_annotation/ to get the relative path
            rel_path = parsed_path.path[len('/sign_annotation/'):]
            # Construct the absolute path on the server
            file_path = os.path.join(SIGN_ANNOTATION_DIR, rel_path)

            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                if file_path.endswith('.npz'):
                    self.send_header('Content-type', 'application/octet-stream')
                    # NPZ files can be overwritten by Save All Changes, so never cache these
                    self._no_cache()
                elif file_path.lower().endswith(('.jpg', '.jpeg')):
                    self.send_header('Content-type', 'image/jpeg')
                    # reference images don't change — let the browser cache them
                elif file_path.lower().endswith('.png'):
                    self.send_header('Content-type', 'image/png')
                    # reference images don't change — let the browser cache them
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_error(404, "File not found")
            return

        # Default: Serve from current directory (website files: app.js, index.html, GLB models, etc.)
        # These are now cacheable too, since we no longer force no-store globally.
        super().do_GET()

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)

        # 4. API: Save modified NPZ
        if parsed_path.path == '/api/save_file':
            query = urllib.parse.parse_qs(parsed_path.query)
            folder = query.get('folder', [''])[0]
            filename = query.get('filename', [''])[0]

            if not folder or not filename:
                self.send_error(400, "Missing folder or filename")
                return

            base_name = folder.replace('_corrected', '')
            target_folder = base_name + '_corrected'

            target_dir = os.path.join(SIGN_ANNOTATION_DIR, 'corrected', target_folder)
            os.makedirs(target_dir, exist_ok=True)

            target_path = os.path.join(target_dir, filename)

            content_length = int(self.headers['Content-Length'])
            file_data = self.rfile.read(content_length)

            with open(target_path, 'wb') as f:
                f.write(file_data)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self._no_cache()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "path": target_folder}).encode())
            return

        self.send_error(404, "Not Found")


if __name__ == '__main__':
    with ThreadedServer(("", PORT), APIHandler) as httpd:
        print(f'Serving at port {PORT}')
        httpd.serve_forever()
