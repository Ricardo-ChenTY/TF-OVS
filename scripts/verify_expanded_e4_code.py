"""Verify public E4 runtime files against both frozen protocol archives; no GPU."""
import hashlib
import json
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARCHIVES=['proposal_e4_expanded_protocol_20261015.tar.gz','corrclip_e4_expanded_protocol_20261015.tar.gz']

def digest(data):return hashlib.sha256(data).hexdigest()

def main():
    matched={};archives={}
    for name in ARCHIVES:
        path=ROOT/'docs/rebuttal/artifacts'/name
        expected=path.with_suffix(path.suffix+'.sha256').read_text().split()[0]
        assert digest(path.read_bytes())==expected, 'Archive checksum mismatch: '+name
        archives[name]=expected
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                if not member.isfile() or '/code/' not in member.name:continue
                rel=member.name.split('/code/',1)[1]
                if not (rel.endswith('.py') or rel=='python_e4'):continue
                target=rel if rel.startswith('src/') else 'scripts/'+rel
                frozen=tar.extractfile(member).read()
                assert (ROOT/target).read_bytes()==frozen,'Published code differs from frozen runtime: '+target
                matched[target]=digest(frozen)
    assert len(matched)==36, 'Unexpected runtime coverage'
    print(json.dumps(dict(status='passed',published_runtime_files=len(matched),archives=archives),indent=2))
if __name__=='__main__':main()
