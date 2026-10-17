"""Replay all six published classifier evaluations on canonical real MNIST."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from utils.metrics import SimpleCNN

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(data,results):
    if sha(data)!='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1':raise ValueError('Canonical MNIST archive required')
    with np.load(data,allow_pickle=False) as z:
        images=torch.from_numpy(z['x_test']).unsqueeze(1).float()/127.5-1;labels=z['y_test']
    torch.set_num_threads(2);records=[]
    for kind in ['cgan','diffusion']:
        for seed in [101,202,303]:
            folder=Path(results)/f'{kind}-{seed}'
            export=json.loads((folder/'export.json').read_text());metrics=json.loads((folder/'utility.json').read_text())
            if sha(folder/'classifier.pt')!=export['classifier_checkpoint_sha256']:raise ValueError('Classifier checkpoint changed')
            if sha(folder/'test-predictions.npz')!=export['test_predictions_sha256']:raise ValueError('Prediction archive changed')
            model=SimpleCNN();model.load_state_dict(torch.load(folder/'classifier.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
            with torch.no_grad():predictions=torch.cat([model(x).argmax(1) for x in images.split(128)]).numpy()
            with np.load(folder/'test-predictions.npz',allow_pickle=False) as z:np.testing.assert_array_equal(predictions,z['predictions'])
            matrix=np.bincount(labels.astype(np.int64)*10+predictions,minlength=100).reshape(10,10)
            np.testing.assert_array_equal(matrix,metrics['confusion_matrix'])
            correct=int((predictions==labels).sum())
            if correct!=metrics['correct']:raise ValueError('Real-test count differs')
            records.append({'model':kind,'seed':seed,'correct':correct,'total':len(labels),'accuracy':correct/len(labels),'all_predictions_exact':True})
    return {'records':records,'torch':str(torch.__version__),'device':'cpu','scope':'Re-evaluation of six saved synthetic-trained classifiers on canonical real-test images. No generator/classifier retraining, sample regeneration or FID repetition.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--results',type=Path,default=Path('results/comparison-v1'));p.add_argument('--output',type=Path);a=p.parse_args()
    record=verify(a.data,a.results);content=json.dumps(record,indent=2)+'\n'
    if a.output:a.output.write_text(content)
    print(content)
if __name__=='__main__':main()
