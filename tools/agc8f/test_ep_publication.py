"""Compile actual publication call; CPU payload behavior is policy emulation only."""
from pathlib import Path
import hashlib, json, re, subprocess, tempfile

S = Path(__file__).resolve().parents[2]
_tmp = tempfile.TemporaryDirectory(prefix='strata-ep-publication-')
P = B = Path(_tmp.name)
v = (S / 'src/core/verify.cpp').read_text()
k = (S / 'src/kernels/cuda/elementwise.cu').read_text()

def call(text):
    found = re.findall(r'                const int32_t\* layer_res = [\s\S]*?m_seq_, cs\);', text)
    assert len(found) == 1
    return found[0]

actual = call(v)
(P / 'publication_call.inc').write_text(actual + '\n')
predicate = re.search(r'if \((d_res == nullptr \|\| id < 0[^\n]+)\) any_miss = 1;', k).group(1)
assert '!host_expert_dispatch_ && hits_.d_res != nullptr' in actual
assert 'if (!host_expert_dispatch_ && hits.h_res != nullptr' in v
assert 'device_plan_ = !host_expert_dispatch_' in v
assert 'bool host_expert_dispatch_ = false;' in (S / 'include/strata/core/verify.hpp').read_text()
stub = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
const int32_t* received;
void doorbell_publish_res(const float*x,const int32_t*ids,const int32_t*d_res,int n_expert,
 int64_t n,int64_t k,float*out,int32_t*seen,uint32_t*seq,void*){
 received=d_res;bool any_miss=false;
 for(int i=0;i<k;i++){const int32_t id=ids[i];seen[i]=id;if(PRED)any_miss=true;}
 if(any_miss)std::copy(x,x+n,out);++*seq;
}
void publish(bool host_expert_dispatch_,const int32_t*res,int l,int tb,int n,int N,int K,
 const float*xm,const int32_t*ids_,float*m_x_,int32_t*m_ids_,uint32_t*m_seq_){
 struct{const int32_t*d_res;}hits_{res};struct{int n_expert;}g{512};void*cs=nullptr;
 CALL
}
'''.replace('PRED', predicate)
tests = r'''
int main(){int cases=0;constexpr int N=37,K=10,L=3;
 for(bool ep:{false,true})for(bool has_res:{false,true})for(int status:{0,1,2,3})
 for(int T:{1,2,3,4,6})for(int tb:{0,2}){
  std::vector<int32_t>res(48*512,0),ids((tb+T)*K),seen((tb+T)*K+2,-77);
  for(int i=0;i<T*K;i++)ids[tb*K+i]=(i*7)%512;
  if(status==1)res[L*512+ids[tb*K+T*K-1]]=-1;
  if(status==2)ids[tb*K+T*K-1]=-1;
  if(status==3)ids[tb*K+T*K-1]=512;
  std::vector<float>x(T*N),out((tb+T)*N+2,-987.0f);uint32_t seq=91;
  for(int pass=0;pass<2;pass++){
   if(pass)std::rotate(ids.begin()+tb*K,ids.begin()+tb*K+1,ids.begin()+(tb+T)*K);
   for(int i=0;i<T*N;i++)x[i]=float(pass*65536+i+1);
   publish(ep,has_res?res.data():nullptr,L,tb,T,N,K,x.data(),ids.data(),
           out.data()+1,seen.data()+1,&seq);
   const auto expected=(!ep&&has_res)?res.data()+L*512:nullptr;
   assert(received==expected);assert(seq==uint32_t(92+pass));
   const bool copies=ep||!has_res||status!=0;
   for(int i=0;i<T*N;i++)assert(out[1+tb*N+i]==(copies?x[i]:-987.0f));
   for(int i=0;i<T*K;i++)assert(seen[1+tb*K+i]==ids[tb*K+i]);
   for(int i=0;i<=tb*N;i++)assert(out[i]==-987.0f);
   for(int i=0;i<=tb*K;i++)assert(seen[i]==-77);
   assert(out.back()==-987.0f&&seen.back()==-77);cases++;
  }
 }
 printf("PASS %d host source publication cases; no CUDA execution\n",cases);
}
'''
baseline = actual.replace('!host_expert_dispatch_ && ', '', 1)
results = []
for label, flags, body, negative in [
    ('plain', [], actual, False),
    ('asan-ubsan', ['-fsanitize=address,undefined', '-fno-omit-frame-pointer'], actual, False),
    ('baseline-negative', [], baseline, True),
]:
    f = B / (label + '.cpp')
    f.write_text(stub.replace('CALL', body) + tests)
    compile_cmd = ['clang++', '-std=c++17', '-O1', *flags, str(f), '-o', str(B / label)]
    c = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert c.returncode == 0, c.stderr
    r = subprocess.run([str(B / label)], capture_output=True, text=True)
    assert (r.returncode != 0) == negative, (label, r.returncode, r.stdout, r.stderr)
    results.append(dict(label=label, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr,
                        expected_failure=negative, compile_command=compile_cmd))
(P / 'host-results.json').write_text(json.dumps(dict(results=results, cases_per_positive_run=320,
    publication_call_sha256=hashlib.sha256(actual.encode()+b'\n').hexdigest(),
    limits='Actual verify source call and kernel miss predicate; payload copy is host emulation, not device proof.'), indent=2)+'\n')
print(json.dumps(results, indent=2))
