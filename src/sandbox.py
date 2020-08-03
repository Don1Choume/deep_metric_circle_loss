from pathlib import Path

outpath = Path(__file__)
print(str(outpath.resolve().parents[1]/'data'/'processed'))