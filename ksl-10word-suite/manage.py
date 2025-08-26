# -*- coding: utf-8 -*-
import argparse, subprocess, sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))

def run(cmd):
    print("[RUN]", " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)

def main():
    ap = argparse.ArgumentParser(description="KSL 10-Word Suite CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest-csv", help="CSV(비디오+구간) → JSONL")
    s.add_argument("--csv", required=True)
    s.add_argument("--out_dir", default="data/signs/jsonl")
    s.add_argument("--fps", type=int, default=30)
    s.set_defaults(func=lambda a: run([sys.executable, "src/ingest_signs_csv.py", "--csv", a.csv, "--out_dir", a.out_dir, "--fps", str(a.fps)]))

    s = sub.add_parser("train", help="DTW 템플릿 학습")
    s.add_argument("--data", default="data/signs/jsonl")
    s.add_argument("--out", default="models/sign_dtw.pkl")
    s.add_argument("--T", type=int, default=32)
    s.add_argument("--k", type=int, default=3)
    s.set_defaults(func=lambda a: run([sys.executable, "src/sign_train_dtw.py", "--data", a.data, "--out", a.out, "--T", str(a.T), "--k", str(a.k)]))

    sub.add_parser("demo", help="실시간 데모(카메라)").set_defaults(func=lambda a: run([sys.executable, "src/sign_infer_realtime.py"]))
    sub.add_parser("ui", help="Streamlit UI").set_defaults(func=lambda a: run(["streamlit", "run", "src/ui_streamlit_sign.py"]))

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
