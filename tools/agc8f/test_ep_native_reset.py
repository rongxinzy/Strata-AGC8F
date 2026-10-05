from pathlib import Path
import subprocess,tempfile,json
S=Path(__file__).resolve().parents[2];text=(S/'src/core/native_dense.cpp').read_text()
start=text.index('bool NativeDense::reset(');end=text.rindex('} // namespace strata::core');reset=text[start:end]
stub=r'''
#include <string>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cstdio>
using cudaError_t=int;constexpr int cudaSuccess=0;int live=0,calls=0,fail_call=0;
const char* cudaGetErrorString(int){return "injected";}
int cudaFree(void* p){++calls;--live;free(p);return calls==fail_call?1:0;}
struct NativeDense{void* scratch_=nullptr;std::vector<void*> weights_;size_t bytes_=0;
bool reset(std::string&);~NativeDense(){std::string err;reset(err);}
void seed(){scratch_=malloc(8);weights_={malloc(8),malloc(8)};live+=3;bytes_=24;}};
'''
test=r'''
int main(){std::string err;int checks=0;{
NativeDense d;assert(d.reset(err)&&calls==0);checks++;
d.seed();assert(d.reset(err));assert(live==0&&d.scratch_==nullptr&&d.weights_.empty()&&d.bytes_==0);checks+=2;
int before=calls;assert(d.reset(err)&&calls==before);checks++;
for(int f=1;f<=3;f++){d.seed();fail_call=calls+f;assert(!d.reset(err));assert(err=="native dense reset: injected");
assert(live==0&&d.scratch_==nullptr&&d.weights_.empty()&&d.bytes_==0);checks+=3;fail_call=0;}
d.seed();assert(d.reset(err));checks++;}
assert(live==0);checks++;printf("PASS %d actual native reset checks\n",checks);}
'''
results=[]
with tempfile.TemporaryDirectory(prefix='strata-reset-') as tmp:
 f=Path(tmp)/'reset.cpp';f.write_text(stub+reset+test)
 for label,flags in [('plain',[]),('asan-ubsan',['-fsanitize=address,undefined','-fno-omit-frame-pointer'])]:
  cmd=['clang++','-std=c++17','-O1',*flags,str(f),'-o',tmp+'/reset-'+label]
  p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0,p.stderr
  p=subprocess.run([tmp+'/reset-'+label],capture_output=True,text=True);assert p.returncode==0,p.stderr
  results.append({'mode':label,'stdout':p.stdout})
print(json.dumps({'reset':results},indent=2))
