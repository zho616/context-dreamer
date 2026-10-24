"""Download official nuPlan log/map archives with resumable, verified S3 chunks."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import zipfile
import xml.etree.ElementTree as ET
import requests

BASE='https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/'
PART=8*1024*1024
NAMES=['nuplan-maps-v1.1.zip','nuplan-v1.1_mini.zip']

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
 p.add_argument('--archives',nargs='+',default=NAMES,help='Official log/map ZIP basenames; sensor_blobs are excluded')
 a=p.parse_args();names=a.archives
 root=a.root.resolve();root.mkdir(parents=True,exist_ok=True);archives=root/'archives';archives.mkdir(exist_ok=True)
 r=requests.get(BASE,params={'list-type':'2','prefix':'public/nuplan-v1.1/','delimiter':'/'},timeout=60);r.raise_for_status();xml=ET.fromstring(r.text);ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
 objects={Path(x.find('s:Key',ns).text).name:{'key':x.find('s:Key',ns).text,'size':int(x.find('s:Size',ns).text),'etag':x.find('s:ETag',ns).text.strip('"')}for x in xml.findall('s:Contents',ns)}
 for name in names:
  if name not in objects or not name.endswith('.zip')or'sensor'in name:p.error(f'Not an official log/map ZIP: {name}')
 manifest={'source':BASE,'files':{n:objects[n]for n in names},'sensor_blobs':False,'created_unix':time.time()}
 if (root/'download_manifest.json').exists():
  previous=json.loads((root/'download_manifest.json').read_text())
  for name,old in previous.get('files',{}).items():
   if name not in manifest['files']:manifest['files'][name]=old
  for name in names:
   old=previous.get('files',{}).get(name,{})
   if old.get('etag')==objects[name]['etag']:
    for field in ['sha256','etag_verified']:
     if field in old:manifest['files'][name][field]=old[field]
 (root/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
 license=requests.get(BASE+'LICENSE',timeout=60);license.raise_for_status();(root/'LICENSE.nuplan.txt').write_text(license.text)
 for name in names:
  meta=objects[name];dest=archives/name;chunkdir=archives/(name+'.parts');chunkdir.mkdir(exist_ok=True)
  parts=(meta['size']+PART-1)//PART;started=time.monotonic()
  def fetch(i):
   start=i*PART;end=min(meta['size'],start+PART)-1;size=end-start+1;path=chunkdir/f'{i:05d}'
   if path.exists()and path.stat().st_size==size:return i
   for attempt in range(6):
    try:
     rr=requests.get(BASE+meta['key'],headers={'Range':f'bytes={start}-{end}'},timeout=(30,180))
     rr.raise_for_status();assert rr.status_code==206 and rr.headers['Content-Range'].startswith(f'bytes {start}-{end}/'),rr.headers
     assert len(rr.content)==size
     tmp=path.with_suffix('.tmp');tmp.write_bytes(rr.content);tmp.replace(path);return i
    except Exception:
     if attempt==5:raise
     time.sleep(min(2**attempt,20))
  if not dest.exists():
   with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers)as pool:
    for done,future in enumerate(concurrent.futures.as_completed([pool.submit(fetch,i)for i in range(parts)]),1):
     future.result()
     if done%16==0 or done==parts:
      progress={'file':name,'completed_parts':done,'total_parts':parts,'elapsed_seconds':round(time.monotonic()-started,1)}
      (root/'download_progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
   md5s=[];sha=hashlib.sha256();tmp=dest.with_suffix('.assembling')
   with tmp.open('wb')as out:
    for i in range(parts):
     b=(chunkdir/f'{i:05d}').read_bytes();md5s.append(hashlib.md5(b).digest());sha.update(b);out.write(b)
   assert tmp.stat().st_size==meta['size']
   calculated=hashlib.md5(b''.join(md5s)).hexdigest()+f'-{parts}'
   assert calculated==meta['etag'],(calculated,meta['etag'])
   tmp.replace(dest);manifest['files'][name].update(sha256=sha.hexdigest(),etag_verified=True)
   (root/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
  target=root/'dataset';target.mkdir(exist_ok=True)
  marker=root/(name+'.extracted.json')
  if not marker.exists():
   with zipfile.ZipFile(dest)as z:
    for info in z.infolist():
     path=target/info.filename
     if not path.resolve().is_relative_to(target.resolve()):raise RuntimeError('Unsafe ZIP path')
    bad=z.testzip();assert bad is None,bad
    z.extractall(target)
    marker.write_text(json.dumps({'members':len(z.infolist()),'top_levels':sorted({Path(x.filename).parts[0]for x in z.infolist()})},indent=2))
   print('Extracted',name,flush=True)
  # Chunks are temporary duplicates of the verified archive, safe to remove.
  if manifest['files'][name].get('etag_verified'):
   for part in chunkdir.iterdir():part.unlink()
   chunkdir.rmdir()
 print('READY',root,flush=True)

if __name__=='__main__':main()
