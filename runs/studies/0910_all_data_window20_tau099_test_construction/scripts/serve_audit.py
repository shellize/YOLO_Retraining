from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parents[4]
STUDY_RELATIVE = Path("runs/studies/0910_all_data_window20_tau099_test_construction")
PAGES = {
    "clusters": STUDY_RELATIVE
    / "result/global_temporal_cluster_preview_label_aware/cluster_preview.html",
    "pairs": STUDY_RELATIVE
    / "result/global_post_dedup_similarity_label_aware/top_similar_pairs.html",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the local audit HTML so browsers can load dataset images."
    )
    parser.add_argument("--page", choices=sorted(PAGES), default="clusters")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Local port. The default 0 asks Windows to choose an available port.",
    )
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    page = PROJECT_ROOT / PAGES[args.page]
    if not page.is_file():
        raise FileNotFoundError(f"audit page does not exist: {page}")

    handler = partial(SimpleHTTPRequestHandler, directory=str(PROJECT_ROOT))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        port = server.server_address[1]
        page_url = quote(PAGES[args.page].as_posix(), safe="/")
        url = f"http://127.0.0.1:{port}/{page_url}"
        print(f"Audit page: {url}", flush=True)
        print("Keep this window open while reviewing. Press Ctrl+C to stop.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nAudit server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
