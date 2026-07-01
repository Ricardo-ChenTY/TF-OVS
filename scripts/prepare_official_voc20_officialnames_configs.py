from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "third_party" / "official_methods"
ARTIFACT_ROOT = ROOT / "runs" / "artifacts" / "official_predictions"
METHODS = ["SCLIP", "NACLIP", "ResCLIP", "ProxyCLIP", "CorrCLIP"]


def main() -> None:
    for method in METHODS:
        cfg_dir = OFFICIAL / method / "configs"
        src = cfg_dir / "cfg_tfovos_voc20.py"
        dst = cfg_dir / "cfg_tfovos_voc20_officialnames.py"
        if not src.exists():
            print(f"[officialnames] skip missing {src}")
            continue
        text = src.read_text(encoding="utf-8")
        text = text.replace("cls_tfovos_voc20.txt", "cls_voc20.txt")
        old_out = str(ARTIFACT_ROOT / method.lower() / "voc20")
        new_out = str(ARTIFACT_ROOT / method.lower() / "voc20_officialnames")
        text = text.replace(old_out, new_out)
        dst.write_text(text, encoding="utf-8")
        print(f"[officialnames] wrote {dst}")


if __name__ == "__main__":
    main()
