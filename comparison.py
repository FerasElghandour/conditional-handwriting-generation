"""Fixed multi-seed cGAN/DDPM training, batched sampling and utility evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from model.cgan import Generator
from model.diffusion import ConditionalUNet
from training.train_diffusion import sample_images

ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def sample_checkpoint(kind,checkpoint,count,batch_size,seed,output,device):
    if count<1 or batch_size<1 or Path(output).exists():raise ValueError('Positive sample budget and new output directory required')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True);meta=ck['metadata']
    if kind=='cgan':model=Generator(z_dim=meta.get('z_dim',100))
    elif kind=='diffusion':model=ConditionalUNet(base_channels=meta.get('base_channels',32),time_dim=meta.get('time_dim',128))
    else:raise ValueError('Unknown model')
    model.load_state_dict(ck['model_state'],strict=True);model.to(device).eval()
    chunks=[];labels=torch.arange(count)%10;torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    with torch.no_grad():
        for start in range(0,count,batch_size):
            y=labels[start:start+batch_size].to(device);torch.manual_seed(seed+start)
            if kind=='cgan':images=model(torch.randn(len(y),model.z_dim,device=device),y)
            else:images,_=sample_images(model,device,len(y),class_labels=y,timesteps=meta['timesteps'],schedule=meta['schedule'],seed=seed+start)
            chunks.append(images.detach().cpu())
            if (start//batch_size+1)%10==0:print(f'{kind}: sampled {min(start+batch_size,count)}/{count}',flush=True)
    images=torch.cat(chunks)
    if not torch.isfinite(images).all() or images.min() < -1.00001 or images.max() > 1.00001:raise ValueError('Invalid generator output')
    output=Path(output);output.mkdir(parents=True)
    np.savez_compressed(output/'samples.npz',images=images.numpy(),labels=labels.numpy())
    pixels=images[:100,0].add(1).mul(127.5).round().clamp(0,255).byte().numpy()
    grid=np.concatenate([np.concatenate(pixels[i:i+10],axis=1) for i in range(0,len(pixels),10)],axis=0)
    Image.fromarray(grid).save(output/'grid.png')
    record={'model':kind,'samples':count,'batch_size':batch_size,'seed':seed,'chunk_seed':'seed+first output index',
            'device':device,'seconds':time.perf_counter()-started,'checkpoint_sha256':sha(checkpoint),
            'samples_sha256':sha(output/'samples.npz'),'class_counts':torch.bincount(labels,minlength=10).tolist(),
            'range':[float(images.min()),float(images.max())],'torch':str(torch.__version__)}
    dump(output/'sampling.json',record)
    return record

def summarize(records,protocol):
    expected={(m,s) for m in protocol['models'] for s in protocol['seeds']}
    indexed={(r['model'],r['seed']):r for r in records}
    if len(indexed)!=len(records) or set(indexed)!=expected:raise ValueError('Complete, unique model/seed pairs required')
    for r in records:
        if r['classifier_seed']!=r['seed']+protocol['classifier']['seed_offset']:raise ValueError('Classifier seed mismatch')
        if r['synthetic_rows']!=protocol['sampling']['count'] or r['test_rows']!=10000:raise ValueError('Comparison sample count mismatch')
        if r['epochs']!=protocol['classifier']['epochs']:raise ValueError('Classifier budget mismatch')
    stats={}
    for model in protocol['models']:
        values=np.array([indexed[(model,s)]['accuracy'] for s in protocol['seeds']])
        stats[model]={'n':len(values),'mean_accuracy':float(values.mean()),'median_accuracy':float(np.median(values)),
                      'sample_sd':float(values.std(ddof=1)),'all_accuracies':values.tolist()}
    pairs=[{'seed':s,'diffusion_minus_cgan':indexed[('diffusion',s)]['accuracy']-indexed[('cgan',s)]['accuracy']} for s in protocol['seeds']]
    return {'models':stats,'paired_differences':pairs,'records':records,'limits':protocol['limits'],
            'interpretation':'Three pipeline seed pairs are descriptive; no inferential superiority claim.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['train','sample','evaluate','summarize','all'],default='all');a=p.parse_args()
    protocol=json.loads((ROOT/'COMPARISON_PROTOCOL.json').read_text());a.data=a.data.resolve();a.out=a.out.resolve()
    if sha(a.data)!=protocol['data_sha256']:raise ValueError('Canonical MNIST archive required')
    files=[ROOT/'train.py',ROOT/'evaluate.py',Path(__file__),ROOT/'COMPARISON_PROTOCOL.json',*sorted((ROOT/'model').glob('*.py')),*sorted((ROOT/'training').glob('*.py')),*sorted((ROOT/'utils').glob('*.py'))]
    hashes={str(f.relative_to(ROOT)):sha(f) for f in files}
    if (a.out/'manifest.json').exists():
        manifest=json.loads((a.out/'manifest.json').read_text())
        if manifest['source_sha256']!=hashes or manifest['protocol']!=protocol:raise ValueError('Source/protocol changed; use a new campaign directory')
    else:
        if a.out.exists() and any(a.out.iterdir()):raise ValueError('New campaign directory required')
        a.out.mkdir(parents=True,exist_ok=True)
        dump(a.out/'manifest.json',{'protocol':protocol,'source_sha256':hashes,'data_sha256':sha(a.data),
                                  'started_unix':time.time(),'python':platform.python_version(),'platform':platform.platform(),'torch':str(torch.__version__)})
    def execute(command,log):
        with log.open('w') as f:subprocess.run([sys.executable,*map(str,command)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    if a.stage in ['train','all']:
        def train_kind(kind):
            spec=protocol['models'][kind]
            for seed in protocol['seeds']:
                run=a.out/f'{kind}-{seed}';run.mkdir(exist_ok=True);train=run/'training'
                if (train/'run.json').exists():continue
                if train.exists():raise ValueError('Incomplete training detected; continue with train.py --resume in a separate folder and preserve its contract')
                execute(['train.py','--data',a.data,'--model',kind,'--epochs',spec['epochs'],'--batch-size',spec['batch_size'],'--seed',seed,'--device',spec['device'],'--output',train],run/'training.log')
                print(f'Completed {kind} training, seed {seed}',flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(train_kind,protocol['models']))
    if a.stage in ['sample','all']:
        for kind in protocol['models']:
            for seed in protocol['seeds']:
                run=a.out/f'{kind}-{seed}'
                if (run/'sampling/sampling.json').exists():continue
                name='cgan_generator_epochfinal.pt' if kind=='cgan' else 'diffusion_unet_epochfinal.pt'
                sample_checkpoint(kind,run/'training'/name,protocol['sampling']['count'],protocol['sampling']['batch_size'],seed+protocol['sampling']['seed_offset'],run/'sampling',protocol['sampling']['device'])
    if a.stage in ['evaluate','all']:
        for kind in protocol['models']:
            for seed in protocol['seeds']:
                run=a.out/f'{kind}-{seed}'
                if (run/'utility/metrics.json').exists():continue
                execute(['evaluate.py','--data',a.data,'--samples',run/'sampling/samples.npz','--epochs',protocol['classifier']['epochs'],'--seed',seed+protocol['classifier']['seed_offset'],'--output',run/'utility'],run/'utility.log')
    if a.stage in ['summarize','all']:
        records=[]
        for kind in protocol['models']:
            for seed in protocol['seeds']:
                run=a.out/f'{kind}-{seed}';record=json.loads((run/'utility/metrics.json').read_text());record.update(model=kind,seed=seed)
                matrix=np.asarray(record['confusion_matrix']);record['per_class_recall']=(matrix.diagonal()/matrix.sum(axis=1)).tolist()
                record['sampling']=json.loads((run/'sampling/sampling.json').read_text());records.append(record)
        dump(a.out/'summary.json',summarize(records,protocol));print('Complete paired comparison recorded.',flush=True)

if __name__=='__main__':main()
