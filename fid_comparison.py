"""Evaluate all declared campaign samples with one shared real-data FID reference."""
import argparse,hashlib,json,platform,time
from importlib import metadata
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

class MPSFeatures(torch.nn.Module):
    num_features=2048
    def __init__(self,extractor):super().__init__();self.extractor=extractor.to('mps')
    @torch.no_grad()
    def forward(self,images):return self.extractor(images.to('mps')).cpu()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--weights',type=Path);p.add_argument('--wait-seconds',type=int,default=10800);a=p.parse_args()
    protocol=json.loads((ROOT/'FID_PROTOCOL.json').read_text());campaign=json.loads((a.campaign/'manifest.json').read_text())
    if sha(a.data)!=campaign['data_sha256'] or a.out.exists():raise ValueError('Canonical data and a new FID output directory required')
    a.out.mkdir(parents=True);torch.hub.set_dir(str(a.out/'torch-cache'));torch.set_num_threads(2)
    manifest={'protocol':protocol,'source_sha256':{f.name:sha(f) for f in [Path(__file__),ROOT/'FID_PROTOCOL.json']},
              'campaign_manifest_sha256':sha(a.campaign/'manifest.json'),'data_sha256':sha(a.data),
              'python':platform.python_version(),'versions':{n:metadata.version(n) for n in ['torch','torchvision','torchmetrics','torch-fidelity','numpy','scipy']},'status':'started','started_unix':time.time()}
    dump(a.out/'manifest.json',manifest)
    from torchmetrics.image.fid import FrechetInceptionDistance
    if not torch.backends.mps.is_available():raise ValueError('This reference protocol requires the qualified MPS feature route')
    reference=FrechetInceptionDistance(feature=2048,normalize=True,feature_extractor_weights_path=str(a.weights) if a.weights else None)
    with np.load(a.data,allow_pickle=False) as z:real=torch.from_numpy(z['x_test'][:protocol['real_samples']]).unsqueeze(1).float()/255
    fixture=(real[:8].repeat(1,3,1,1)*255).byte()
    with torch.no_grad():cpu_features=reference.inception(fixture)
    features=MPSFeatures(reference.inception)
    with torch.no_grad():gpu_features=features(fixture)
    torch.testing.assert_close(cpu_features,gpu_features,atol=.002,rtol=.002)
    manifest['feature_qualification']={'real_images':8,'absolute_tolerance':.002,'relative_tolerance':.002,
          'max_absolute_difference':float((cpu_features-gpu_features).abs().max()),'passed':True}
    metric=FrechetInceptionDistance(feature=features,normalize=False,reset_real_features=False)
    if a.weights:manifest['feature_weight_sha256']={a.weights.name:sha(a.weights)}
    def update(images,real):
        for start in range(0,len(images),32):
            metric.update((images[start:start+32].repeat(1,3,1,1)*255).byte(),real=real)
            if (start//32+1)%25==0:print(f'FID {"real" if real else "fake"}: {min(start+32,len(images))}/{len(images)}',flush=True)
    update(real,True)
    if not a.weights:manifest['feature_weight_sha256']={f.name:sha(f) for f in (a.out/'torch-cache').rglob('*.pth')}
    dump(a.out/'manifest.json',manifest);records=[]
    for kind in campaign['protocol']['models']:
        for seed in campaign['protocol']['seeds']:
            samples=a.campaign/f'{kind}-{seed}/sampling/samples.npz';deadline=time.monotonic()+a.wait_seconds
            while not (samples.parent/'sampling.json').exists():
                if time.monotonic()>deadline:raise TimeoutError('Campaign sampling did not finish within declared wait budget')
                time.sleep(5)
            sampling=json.loads((samples.parent/'sampling.json').read_text())
            if sampling['samples_sha256']!=sha(samples):raise ValueError('Sampling archive fingerprint mismatch')
            with np.load(samples,allow_pickle=False) as z:
                fake=torch.from_numpy(z['images'][:protocol['fake_samples']]).float().clamp(-1,1).add(1).div(2)
            if len(fake)!=protocol['fake_samples']:raise ValueError('Fixed fake sample count required')
            started=time.perf_counter();update(fake,False);score=float(metric.compute());metric.reset()
            record={'model':kind,'seed':seed,'fid_2048':score,'real_samples':len(real),'fake_samples':len(fake),
                    'sample_archive_sha256':sha(samples),'checkpoint_sha256':sampling['checkpoint_sha256'],'wall_seconds':time.perf_counter()-started}
            records.append(record);dump(a.out/'results.json',records);print(json.dumps(record),flush=True)
    aggregate={}
    for kind in campaign['protocol']['models']:
        values=np.array([r['fid_2048'] for r in records if r['model']==kind]);aggregate[kind]={'n':len(values),'mean':float(values.mean()),'median':float(np.median(values)),'sample_sd':float(values.std(ddof=1))}
    dump(a.out/'summary.json',{'models':aggregate,'records':records,'limits':protocol['limits']});manifest.update(status='completed',finished_unix=time.time());dump(a.out/'manifest.json',manifest)

if __name__=='__main__':main()
