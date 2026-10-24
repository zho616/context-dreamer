"""Extract a small CRC-checked log subset from cached official Mini ZIP chunks."""
import argparse,io,json,os,shutil,time,zipfile
from pathlib import Path
import requests
PART=8*1024*1024
class CachedArchive(io.RawIOBase):
 def __init__(self,root,meta):self.root=root;self.meta=meta;self.pos=0;self.cache=root/'archives'/(Path(meta['key']).name+'.parts');self.cache.mkdir(exist_ok=True)
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.meta['size']+offset
  return self.pos
 def read(self,n=-1):
  if n<0:n=self.meta['size']-self.pos
  n=min(n,self.meta['size']-self.pos);pieces=[]
  while n>0:
   index=self.pos//PART;offset=self.pos%PART;size=min(PART,self.meta['size']-index*PART);path=self.cache/f'{index:05d}'
   if not path.exists()or path.stat().st_size!=size:
    start=index*PART;end=start+size-1
    r=requests.get('https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/'+self.meta['key'],headers={'Range':f'bytes={start}-{end}'},timeout=(30,180));r.raise_for_status();assert r.status_code==206 and len(r.content)==size
    tmp=path.with_suffix(f'.priority{os.getpid()}');tmp.write_bytes(r.content);tmp.replace(path)
   take=min(n,size-offset)
   with path.open('rb')as f:f.seek(offset);pieces.append(f.read(take))
   self.pos+=take;n-=take
  return b''.join(pieces)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--logs',type=int,default=8);a=p.parse_args()
 root=a.root.resolve();manifest=json.loads((root/'download_manifest.json').read_text());meta=manifest['files']['nuplan-v1.1_mini.zip']
 output=root/'priority_mini';output.mkdir(exist_ok=True)
 with zipfile.ZipFile(CachedArchive(root,meta))as archive:
  logs=sorted([x for x in archive.infolist()if x.filename.endswith('.db')],key=lambda x:x.header_offset)
  records=[]
  for info in logs[:a.logs]:
   target=output/Path(info.filename).name
   if not target.exists():
    temp=target.with_suffix('.extracting')
    with archive.open(info)as source,temp.open('wb')as dest:shutil.copyfileobj(source,dest,PART)
    assert temp.stat().st_size==info.file_size;temp.replace(target)
   records.append({'log':target.name,'member':info.filename,'size':info.file_size,'crc':info.CRC,'archive_offset':info.header_offset})
   print('Priority log extracted',target.name,flush=True)
  (output/'manifest.json').write_text(json.dumps({'source_archive':manifest['source']+meta['key'],'selection':f'First{a.logs} DB members in archive order for integration smoke only; no driving-outcome filtering','total_archive_logs':len(logs),'all_log_names':[Path(x.filename).name for x in logs],'logs':records},indent=2))
 print('PRIORITY_LOGS_READY',output,flush=True)
if __name__=='__main__':main()
