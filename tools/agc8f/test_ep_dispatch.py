from pathlib import Path
import subprocess,json,os
import tempfile
S=Path(__file__).resolve().parents[2]
_tmp=tempfile.TemporaryDirectory(prefix='strata-ep-regression-');B=Path(_tmp.name)
t=(S/'src/core/expert_source.cpp').read_text();a=t.index('namespace {\n// the verify window');b=t.index('\nvoid expert_hit_run(',a)
actual=t[a:b]
stub=r'''
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <array>
namespace strata::kernels::cpu {
constexpr int H=32,MAXT=8,kNativeActBytes=32;struct Act{};struct Fmt{int gu_type=18;};
struct ExpertLayout{bool native=true;std::vector<Fmt>fmt=std::vector<Fmt>(48);uint64_t blob_bytes(int)const{return 8;}};
const ExpertLayout& expert_layout(){static ExpertLayout l;return l;}
bool q2_native_kernels(int){return false;}
void act_quant_any(const float*,int,Act&){}void act_quant_q8_1(const float*,int,Act&){}
void native_quant_act(const Fmt&,const float*,uint8_t*){}
struct ExpertJobMulti{const uint8_t*blob=nullptr;int nt=0;Act*act[8];uint8_t*nact[8];float*out[8];};
}
namespace strata::core {
using namespace strata::kernels::cpu;
struct Source{void begin_layer(int,const int32_t*,int64_t){}bool pcie_layer(int){return false;}
bool pinned(int,int){return false;}const uint8_t*blob(int,int){static uint8_t b[8];return b;}
const uint8_t*device_alias(int,int){return nullptr;}void prefetch(int,const int64_t*,int64_t){}};
struct Lookahead{void submit(int,const float*,int64_t,const int32_t*){}};
struct Peer{bool has(int,int){return false;}bool launch(int,const float*,const int32_t*,int64_t,int64_t,const int32_t*,std::string&,float*){return true;}
bool launched_direct(){return false;}bool finish(float*,std::string&){return true;}};
struct Pool{void run_split_multi_native(const Fmt&,ExpertJobMulti*,int){}void run_split_multi(ExpertJobMulti*,int){}};
struct Remote{int r=0,begins=0,drains=0,publishes=0,fail_begin=0,fail_finish=0;bool on=true,pending=false;
std::vector<int>owned;bool ep_owns(int l,int e,int64_t t,int64_t k){return on&&l==0&&t>=4&&k==10&&e%4==r;}
bool optimized_decode(){return false;}bool owns(int64_t i){return i<(int64_t)owned.size()&&owned[i];}
bool begin(int l,const float*,const int32_t*ids,int64_t t,int64_t k,const int32_t*kind,const int32_t*,std::string&e){
 ++begins;pending=true;owned.assign(t*k,0);for(int i=0;i<t*k;i++)owned[i]=kind[i]==2&&ep_owns(l,ids[i],t,k);
 if(fail_begin){--fail_begin;e="begin failed";return false;}return true;}
bool finish(float*out,std::string&e){if(!pending)return true;++drains;pending=false;
 if(fail_finish){--fail_finish;e="finish failed";return false;}
 if(out){++publishes;for(size_t i=0;i<owned.size();i++)if(owned[i])for(int h=0;h<H;h++)out[i*H+h]=float(r*1000+i);}
 return true;}
};
struct GpuPlanSink{int cap=128,staging_cap=64,pcie_mode=0;uint64_t staging=0;
unsigned long long ptr[128],ptr2[128];int32_t start[129],start2[129],dst[128],tok[128],counts[3];
void*ctx=nullptr;void(*publish)(void*)=nullptr;void(*fetch)(void*,const uint8_t**,int,size_t)=nullptr;};
struct ExpertDispatch{bool failed=false,split_rows=false;const char*fail=nullptr;int64_t fail_layer=0,fail_expert=0,layers=0;
int64_t n_expert=512;Source*src=nullptr;Lookahead*lookahead=nullptr;Peer*peer=nullptr;Pool*pool=nullptr;
GpuPlanSink*plan=nullptr;Remote*remote[7]={};int remote_count=0,pcie_num=0;
int32_t*host_res=nullptr;const uint8_t*cache_base=nullptr;const uint64_t*cache_slot_off=nullptr;uint64_t cache_blob=8;
std::vector<float>usage;std::vector<Act>act_multi;std::vector<uint8_t>nact_multi;
std::vector<int16_t>job_of;std::vector<ExpertJobMulti>jobs_multi;
int64_t pcie_experts=0,cache_hits=0,offload_entries=0,peer_entries=0,cache_refused=0,missing=0,multi_entries=0,multi_misses=0,experts=0;
double ms_plan=0,ms_actq=0,ms_jobs=0,ms_run=0;};
'''
test=r'''
}
int main(){using namespace strata::core;int checks=0;Source src;Pool pool;Remote r[3];for(int j=0;j<3;j++)r[j].r=j+1;
int32_t ids[80],res[48*512];std::fill(res,res+48*512,0);float x[8*H]={},out[80*H];uint8_t cache[512*8]={};
for(int t=0;t<8;t++)for(int k=0;k<10;k++)ids[t*10+k]=(k*7+t)%32;
for(int mode=0;mode<8;mode++){
 ExpertDispatch d;GpuPlanSink p;d.src=&src;d.pool=&pool;d.host_res=res;d.cache_base=cache;d.plan=&p;d.remote_count=3;
 for(int j=0;j<3;j++){r[j].on=mode!=0;r[j].begins=r[j].drains=r[j].publishes=0;d.remote[j]=r+j;}
 std::fill(out,out+80*H,-5);if(mode==2)r[1].fail_begin=1;if(mode==3)r[1].fail_finish=1;
 if(mode==4)ids[79]=-1;if(mode==5)ids[79]=512;if(mode==6)d.plan=nullptr;
 const int T=mode==7?3:8;expert_pool_dispatch_multi(d,x,ids,T,10,out);
 assert(d.failed==(mode==2||mode==3||mode==4||mode==5));checks++;
 for(int j=0;j<3;j++){assert(!r[j].pending);checks++;if(mode==2){assert(r[j].begins==(j<=1?1:0));assert(r[j].drains==(j<=1?1:0));}
 else {assert(r[j].begins==1);assert(r[j].drains==1);}checks+=2;}
 if(mode==1){
  std::vector<int>seen(80);for(int g=0;g<p.counts[0];g++)for(int q=p.start[g];q<p.start[g+1];q++){
   const int rank=p.dst[q];assert(ids[rank]%4==0&&p.tok[q]==rank/10);seen[rank]++;checks++;}
  for(int i=0;i<80;i++){assert(seen[i]==(ids[i]%4==0?1:0));
   if(ids[i]%4)assert(out[i*H]==float((ids[i]%4)*1000+i));checks+=2;}
 }
 if(mode==0||mode==7){assert(p.counts[1]==T*10);for(int i=0;i<T*10;i++)assert(out[i*H]==0);checks++;}
 ids[79]=(9*7+7)%32;
}
printf("PASS %d actual dispatch checks (8 scenarios)\n",checks);}
'''
f=B/'actual_dispatch.cpp';f.write_text(stub+actual+test);results=[]
for name,flags in [('plain',[]),('asan-ubsan',['-fsanitize=address,undefined','-fno-omit-frame-pointer'])]:
 cmd=['clang++','-std=c++17','-O1',*flags,str(f),'-o',str(B/name)]
 q=subprocess.run(cmd,capture_output=True,text=True);assert q.returncode==0,q.stderr
 q=subprocess.run([str(B/name)],capture_output=True,text=True,env={**os.environ,"STRATA_DEC_BATCH":"0"});assert q.returncode==0,q.stderr
 results.append({'mode':name,'stdout':q.stdout,'source':'verbatim expert_pool_dispatch_multi + window constants','scenarios':8})
print(json.dumps(results,indent=2))
