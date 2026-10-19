import json
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e4_full_common import confusion_matrix,quality,open_db,validate_prefix,sha,pack_confusion,finish_job

def test_void_and_unknown_predictions_count_as_false_negatives():
    cm=confusion_matrix(np.array([[0,0,1,255]]),np.array([[0,255,0,1]]),2)
    assert cm.tolist()==[[1,0,1],[1,0,0]]
    assert quality(cm)['miou_percent']==pytest.approx(100/6)
    assert quality(cm)['valid_pixels']==3
    with pytest.raises(ValueError):confusion_matrix(np.array([[17]]),np.array([[0]]),2)
    with pytest.raises(ValueError):confusion_matrix(np.array([[0]]),np.array([[0.5]]),2)

def test_resume_integrity_and_incomplete_finalize(tmp_path):
    rows=[dict(image_id='first'),dict(image_id='second')]
    db=open_db(tmp_path/'test.sqlite',{'identity':'a'})
    prediction=tmp_path/'prediction.png';prediction.write_bytes(b'committed prediction')
    record=dict(prediction_file=prediction.name,prediction_file_sha256=sha(prediction))
    with db:db.execute('INSERT INTO images VALUES (?,?,?,?)',('first',0,json.dumps(record),pack_confusion(np.zeros((2,3)))))
    assert validate_prefix(db,rows,tmp_path)==1
    with pytest.raises(RuntimeError):finish_job(db,rows,tmp_path,{'classes':2})
    with pytest.raises(RuntimeError):open_db(tmp_path/'test.sqlite',{'identity':'changed'})
    with pytest.raises(sqlite3.IntegrityError):
        with db:db.execute('INSERT INTO images VALUES (?,?,?,?)',('first',1,json.dumps(record),b''))
    assert validate_prefix(db,rows,tmp_path)==1
    prediction.write_bytes(b'corrupted')
    with pytest.raises(RuntimeError):validate_prefix(db,rows,tmp_path)
    db.close()

def test_final_result_uses_all_committed_predictions(tmp_path):
    rows=[dict(image_id='a'),dict(image_id='b')]
    db=open_db(tmp_path/'test.sqlite',{'identity':'full'})
    matrices=[np.array([[2,0,0],[0,0,1]]),np.array([[0,1,0],[0,1,0]])]
    for ordinal,row in enumerate(rows):
        rec=dict(online_seconds=ordinal+1,read_to_labels_seconds=ordinal+2,
                 peak_allocated_mib=1024*(ordinal+1),peak_reserved_mib=3072)
        with db:db.execute('INSERT INTO images VALUES (?,?,?,?)',(row['image_id'],ordinal,json.dumps(rec),pack_confusion(matrices[ordinal])))
    result=finish_job(db,rows,tmp_path,{'classes':2})
    assert result['images']==2 and result['online_mean_seconds']==1.5
    assert result['peak_allocated_gib']==2
    assert result['miou_percent']==pytest.approx(50)
    assert result['valid_pixels']==5 and result['rejected_prediction_pixels']==1
    assert np.array_equal(np.load(tmp_path/'confusion.npy'),matrices[0]+matrices[1])
    db.close()

def test_targeted_pairing_detects_a_changed_prediction(tmp_path):
    from PIL import Image
    from run_e4_targeted import audit_pair
    from e4_full_common import save
    out=tmp_path/'jobs/sam_amg_clip/voc20';out.mkdir(parents=True)
    reference=tmp_path/'reference.png';prediction=out/'prediction.png'
    Image.fromarray(np.array([[0,1]],dtype=np.uint8)).save(reference)
    Image.fromarray(np.array([[0,0]],dtype=np.uint8)).save(prediction)
    save(tmp_path/'references.json',{'sam_amg_clip/voc20':{'a':{'path':str(reference),'sha256':sha(reference)}}})
    save(tmp_path/'protocol.json',{'references_sha256':sha(tmp_path/'references.json')})
    (out/'per_image.jsonl').write_text(json.dumps({'image_id':'a','prediction_file':prediction.name,'prediction_file_sha256':sha(prediction)})+'\n')
    result=audit_pair(tmp_path,'sam_amg_clip','voc20')
    assert result['status']=='mismatches_require_review'
    assert result['changed_pixels']==1 and result['exact_matches']==0
