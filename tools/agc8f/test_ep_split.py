from pathlib import Path
import subprocess,json
S=Path(__file__).resolve().parents[2]
import tempfile
_tmp=tempfile.TemporaryDirectory(prefix='strata-ep-test-');P=Path(_tmp.name)
text=(S/'src/program/generate.cpp').read_text()
a=text.index('struct SplitDrive {');b=text.index('/// Layer split across GPUs:',a)
stub=r'''
#include <cstdint>
#include <cassert>
#include <cstdio>
namespace strata::core {struct GpuPlanSink{};struct RemoteExperts{};}
struct Dispatch {strata::core::GpuPlanSink* plan=nullptr;const uint8_t* cache_base=nullptr;const uint64_t* cache_slot_off=nullptr;int pcie_num=0,remote_count=0;strata::core::RemoteExperts* remote[3]={};};
struct Drive {Dispatch d;};
int called=0;
void drive_pool_multi(void*,const float*,const int32_t*,int64_t,int64_t,float*,int64_t){called++;}
'''
test=r'''
int main(){Drive drive;SplitDrive split;split.base=&drive;split.n=8;strata::core::RemoteExperts helpers[7];
for(int s=0;s<8;s++){split.end[s]=(s+1)*6;for(int r=0;r<3;r++)split.ep_remote[s][r]=&helpers[(s+r)%7];}
// OFF does not change legacy pointers or count.
drive.d.remote_count=1;drive.d.remote[0]=&helpers[6];drive_pool_split(&split,nullptr,nullptr,4,10,nullptr,25);assert(drive.d.remote_count==1&&drive.d.remote[0]==&helpers[6]);
split.ep24=true;int checks=1;
// Repeated traversals model first/repeated verify routing; layer48 explicitly clears count.
for(int run=0;run<3;run++)for(int l=0;l<=48;l++){drive_pool_split(&split,nullptr,nullptr,4,10,nullptr,l);assert(drive.d.remote_count==(l<48?3:0));checks++;if(l<48)for(int r=0;r<3;r++){assert(drive.d.remote[r]==split.ep_remote[l/6][r]);checks++;}}
printf("PASS %d actual SplitDrive checks\n",checks);}
'''
f=P/'actual_split_host.cpp';f.write_text(stub+text[a:b]+test);results=[]
for name,flags in [('plain',[]),('sanitized',['-fsanitize=address,undefined'])]:
 cmd=['clang++','-std=c++17',*flags,str(f),'-o',str(P/('split-'+name))]
 subprocess.run(cmd,check=True);r=subprocess.run([str(P/('split-'+name))],check=True,capture_output=True,text=True);results.append({'command':cmd,'stdout':r.stdout})
(P/'split-results.json').write_text(json.dumps(results,indent=2));print(results)
