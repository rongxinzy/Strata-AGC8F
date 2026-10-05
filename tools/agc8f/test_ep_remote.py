from pathlib import Path
import subprocess,json
S=Path(__file__).resolve().parents[2]
import tempfile
_tmp=tempfile.TemporaryDirectory(prefix='strata-ep-test-');P=Path(_tmp.name)
h=(S/'include/strata/core/remote_experts.hpp').read_text();c=(S/'src/core/remote_experts.cpp').read_text()
policy=(S/"include/strata/core/ep24_plan.hpp").read_text()
opt=(S/"include/strata/core/remote_expert_opt.hpp").read_text()
opt="\n".join(l for l in opt.splitlines() if not l.startswith("#include") and l!="#pragma once")
h='\n'.join(l for l in h.splitlines() if not l.startswith('#include') and l!='#pragma once')
c='\n'.join(l for l in c.splitlines() if not l.startswith('#include'))
stub=r'''
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <limits>
#include <exception>
#include <new>
using cudaError_t=int; using cudaStream_t=void*;
constexpr int cudaSuccess=0,cudaStreamNonBlocking=1,cudaHostAllocPortable=1,cudaHostAllocMapped=2,cudaDeviceScheduleSpin=1,cudaDeviceMapHost=2,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
int allocations=0,streams=0,failalloc=0,memcalls=0,failmem=0,throwmem=0,failsync=0,failstream=0,faillaunch=0;
int reduced_calls=0,accumulated_calls=0,reduce_fail=0;
int current=0,verified=0,syncs=0,failcopy=0; size_t available=16ull<<30;
int cudaGetDevice(int*p){*p=current;return 0;} int cudaSetDevice(int p){current=p;return 0;}
const char* cudaGetErrorString(int){return "injected";}
int cudaGetDeviceCount(int*p){*p=8;return 0;}int cudaInitDevice(int,int,int){return 0;}int cudaGetLastError(){if(faillaunch){--faillaunch;return 1;}return 0;}
int cudaMemGetInfo(size_t*a,size_t*b){++memcalls;if(memcalls==throwmem)throw std::bad_alloc();*a=*b=memcalls==failmem?0:available;return 0;}
int cudaStreamSynchronize(void*){++syncs;if(failsync){--failsync;return 1;}return 0;}int cudaStreamCreateWithFlags(void**p,int){if(failstream){--failstream;return 1;}++streams;*p=(void*)1;return 0;}int cudaStreamDestroy(void*){--streams;return 0;}
int cudaFree(void*p){if(p)--allocations;free(p);return 0;}int cudaFreeHost(void*p){return cudaFree(p);}
int cudaMalloc(void**p,size_t n){if(failalloc&&!--failalloc)return 1;*p=calloc(1,n);++allocations;return 0;}int cudaHostAlloc(void**p,size_t n,int){return cudaMalloc(p,n);}
int cudaHostGetDevicePointer(void**p,void*q,int){*p=q;return 0;}
int cudaMemcpyAsync(void*d,const void*s,size_t n,int,void*){if(failcopy&&!--failcopy)return 1;memcpy(d,s,n);return 0;}
namespace strata::kernels::cpu { constexpr int H=32,FF=32,MAXT=8;struct Fmt{int gu_type=18,d_type=20,n_embd=32,n_ff=32;};
struct Layout {bool native=true;size_t max_blob=8;std::vector<Fmt>fmt=std::vector<Fmt>(48);size_t blob_bytes(int)const{return 8;}};Layout& expert_layout(){static Layout l;return l;} }
namespace strata::core {
struct ExpertSource {const uint8_t* blob(int,int){static uint8_t b[8]={};return b;}};
struct ExpertCache {std::vector<std::pair<int,int>> entries;bool primary=false; int lb=0,le=48;
void close(){entries.clear();}bool open_sized(const std::vector<int64_t>&,int64_t,int64_t,std::string&){return true;}
bool open(int64_t,int64_t,int64_t,int64_t,std::string&){return true;}
int slot_of(int64_t l,int64_t e)const{if(primary)return l>=lb&&l<le?e:-1;for(size_t i=0;i<entries.size();i++)if(entries[i]==std::make_pair((int)l,(int)e))return i;return -1;}
int admit(int l,int e){entries.emplace_back(l,e);return entries.size()-1;}
bool fill_slot_blocking(int,const uint8_t*,std::string&,int64_t){return true;}
bool verify_slot(int,const uint8_t*,std::string&,int64_t){++verified;return true;}
const uint8_t* device_slot(int)const{static uint8_t b[8]={};return b;}
int64_t resident()const{return entries.size();}double gib()const{return 0;}};
}
namespace strata::kernels {
size_t moe_hit_grouped_scratch_bytes(int,int,int){return 32;}size_t native_expert_scratch_bytes(int,int){return 32;}
int native_expert_layout(int,int,int,int){return 0;}
void quantize_q8_1_rows(const float*,int64_t,int64_t,uint8_t*,void*){}
void quantize_q8_0_scaled(const float*,uint8_t*,float*,int64_t,void*){}
void native_expert_grouped(int,unsigned long long*,int32_t*,int32_t*,int32_t*,int32_t*,int,int64_t n,uint8_t*,void*,float*out,void*){for(int i=0;i<n*32;i++)out[i]=(float)(i/32+1);}
void moe_grouped_s2(unsigned long long*,int32_t*,int32_t*,int32_t*,int32_t*,int,int64_t,uint8_t*,float*,void*,float*,void*){}
}
'''
test=r'''
int main(){using namespace strata::core;ExpertCache primary;primary.primary=true;ExpertSource src;std::string err;
RemoteExperts rs[3]; std::vector<uint8_t> claimed(48*512);int checks=0;
for(int r=0;r<3;r++){std::vector<std::pair<int32_t,int32_t>> ranked;for(int e=r+1;e<512;e+=4)ranked.emplace_back(0,e);
assert(rs[r].open(r+1,128,48,512,ranked,primary,src,claimed,err,false,true));assert(rs[r].resident()==128);checks+=2;}
assert(verified==387);checks++;
for(int t=1;t<=8;t++)for(int l: {0,1,47})for(int e=0;e<512;e++)for(int r=0;r<3;r++){assert(rs[r].ep_owns(l,e,t,10)==(t>=4&&l==0&&e%4==r+1));checks++;}
float x[8*32]={},out[80*32];int32_t ids[80],kind[80],res[48*512];std::fill(res,res+48*512,0);
for(int variant=0;variant<4;variant++){std::fill(out,out+80*32,-5);for(int i=0;i<80;i++){ids[i]=(i*7+variant)%512;kind[i]=(ids[i]%4)?2:0;}
for(int r=0;r<3;r++)assert(rs[r].begin(0,x,ids,8,10,kind,res,err));
for(int r=0;r<3;r++)assert(rs[r].finish(out,err));
for(int i=0;i<80;i++){assert((out[i*32]>0)==(ids[i]%4!=0));checks++;}}
// OFF keeps every replica and restores ownership with no refill.
for(int r=0;r<3;r++) {
const int before=verified; const auto resident=rs[r].resident();
assert(rs[r].ep_set_active(false));
for(int e=0;e<512;e++){assert(!rs[r].ep_owns(0,e,8,10));checks++;}
assert(rs[r].begin(0,x,ids,8,10,kind,res,err));
for(int i=0;i<80;i++){assert(!rs[r].owns(i));checks++;}
assert(rs[r].finish(out,err));assert(rs[r].ep_set_active(true));
assert(rs[r].resident()==resident && verified==before);
assert(rs[r].ep_owns(0,r+1,8,10));
assert(rs[r].begin(0,x,ids,8,10,kind,res,err));
assert(!rs[r].ep_set_active(false));assert(!rs[r].ep_set_active(true));
assert(rs[r].ep_owns(0,r+1,8,10));assert(rs[r].finish(out,err));checks+=11;
}
// partial enqueue failure retains buffer lease until explicit draining finish.
failcopy=1;assert(!rs[0].begin(0,x,ids,8,10,kind,res,err));assert(!rs[0].begin(0,x,ids,8,10,kind,res,err));assert(!rs[0].ep_set_active(false));checks++;assert(rs[0].finish(nullptr,err));
assert(rs[0].begin(0,x,ids,8,10,kind,res,err));assert(rs[0].finish(out,err));checks+=5;
for(int r=0;r<3;r++) {assert(rs[r].begin(1,x,ids,8,10,kind,res,err));for(int i=0;i<80;i++){assert(!rs[r].owns(i));checks++;}assert(rs[r].finish(out,err));}

// Failed enqueues can be drained with an output argument but must not publish stale rows.
for(int failure=1;failure<=4;failure++) {
std::fill(out,out+80*32,-77);if(failure<=3)failcopy=failure;else faillaunch=1;
assert(!rs[0].begin(0,x,ids,8,10,kind,res,err));assert(!rs[0].finish(out,err));
assert(std::all_of(out,out+80*32,[](float v){return v==-77;}));checks+=3;
}
assert(rs[0].begin(0,x,ids,8,10,kind,res,err));failsync=1;
assert(!rs[0].finish(out,err));assert(!rs[0].ep_set_active(false));
assert(!rs[0].begin(0,x,ids,8,10,kind,res,err));assert(rs[0].finish(nullptr,err));checks+=5;
for(auto t:{INT64_MIN,INT64_MAX,int64_t(0),int64_t(9)})assert(!rs[0].begin(0,x,ids,t,10,kind,res,err));
assert(!rs[0].begin(0,x,ids,8,INT64_MAX,kind,res,err));
assert(!rs[0].begin(0,nullptr,ids,8,10,kind,res,err));assert(!rs[0].begin(0,x,nullptr,8,10,kind,res,err));checks+=7;

// Actual production selection policy: largest blobs, deterministic ascending ties.
auto plan=strata::core::ep24_plan([](int l){return l%6==5?20:10;});
for(int st=0;st<8;st++){assert(plan.layers[st][0]==st*6+5);assert(plan.layers[st][1]==st*6);assert(plan.layers[st][2]==st*6+1);
for(int r=0;r<3;r++){assert(plan.helpers[st][r]>0&&plan.helpers[st][r]!=st);for(int q=0;q<r;q++)assert(plan.helpers[st][q]!=plan.helpers[st][r]);}checks+=12;}
ExpertCache primaries[8];std::vector<const ExpertCache*> by_layer(48);
for(int st=0;st<8;st++){primaries[st].primary=true;primaries[st].lb=6*st;primaries[st].le=6*st+6;for(int l=6*st;l<6*st+6;l++)by_layer[l]=&primaries[st];}
std::vector<std::vector<int>> ranks(7,std::vector<int>(48));
for(int st=0;st<8;st++)for(int r=0;r<3;r++)for(int l:plan.layers[st])ranks[plan.helpers[st][r]-1][l]=r+1;
RemoteExperts helpers[7];std::vector<uint8_t> used(48*512);int before_verify=verified,total_replicas=0;
for(int dev=1;dev<=7;dev++){std::vector<std::pair<int32_t,int32_t>> ranked;
for(int l=0;l<48;l++)if(ranks[dev-1][l])for(int e=ranks[dev-1][l];e<512;e+=4)ranked.emplace_back(l,e);
assert(helpers[dev-1].open(dev,ranked.size(),48,512,ranked,primaries[0],src,used,err,false,true,&ranks[dev-1],&by_layer));total_replicas+=ranked.size();checks++;}
assert(total_replicas==24*384);assert(verified-before_verify==total_replicas+7);checks+=2;
for(int t=1;t<=8;t++)for(int l=-1;l<=48;l++)for(int e=0;e<512;e++)for(int dev=1;dev<=7;dev++){
bool expected=l>=0&&l<48&&t>=4&&ranks[dev-1][l]>0&&e%4==ranks[dev-1][l];
assert(helpers[dev-1].ep_owns(l,e,t,10)==expected);checks++;}
// Repeated full-stage traversals retain rank rows while reusing each physical helper across layers.
for(int repeat=0;repeat<3;repeat++)for(int st=0;st<8;st++)for(int l:plan.layers[st]){
std::fill(out,out+80*32,-5);for(int i=0;i<80;i++){ids[i]=(i*7+repeat)%512;kind[i]=ids[i]%4?2:0;}
for(int dev:plan.helpers[st])assert(helpers[dev-1].begin(l,x,ids,8,10,kind,res,err));
for(int dev:plan.helpers[st])assert(helpers[dev-1].finish(out,err));
for(int i=0;i<80;i++){assert((out[i*32]>0)==(ids[i]%4!=0));checks++;}}
// All seven helpers retain their ranks and allocations across repeated sameboot OFF/ON.
for(int repeat=0;repeat<4;repeat++)for(int dev=1;dev<=7;dev++) {
const auto resident=helpers[dev-1].resident();const int before=verified;
assert(helpers[dev-1].ep_set_active(repeat%2));
for(int l=0;l<48;l++)for(int e=0;e<512;e++) {
bool expected=repeat%2 && ranks[dev-1][l]>0 && e%4==ranks[dev-1][l];
assert(helpers[dev-1].ep_owns(l,e,8,10)==expected);checks++;
}
assert(resident==helpers[dev-1].resident() && verified==before);checks+=2;
}
// Closing and reopening returns to default-active, including an unopened object.
RemoteExperts reset;assert(reset.ep_set_active(false));reset.close();
std::vector<std::pair<int32_t,int32_t>> reset_rank;for(int e=1;e<512;e+=4)reset_rank.emplace_back(0,e);
std::vector<uint8_t> reset_claimed(48*512);
assert(reset.open(1,128,48,512,reset_rank,primary,src,reset_claimed,err,false,true));
assert(reset.ep_owns(0,1,8,10));assert(reset.ep_set_active(false));reset.close();
std::fill(reset_claimed.begin(),reset_claimed.end(),0);
assert(reset.open(1,128,48,512,reset_rank,primary,src,reset_claimed,err,false,true));assert(reset.ep_owns(0,1,8,10));checks+=6;
// API refuses incorrect real-stage residency and capacity; no fallback or cache shrink.
std::vector<std::pair<int32_t,int32_t>> one;for(int e=1;e<512;e+=4)one.emplace_back(47,e);
std::vector<int> one_rank(48);one_rank[47]=1;std::vector<uint8_t> fresh(48*512);RemoteExperts bad;
auto wrong=by_layer;wrong[47]=&primaries[0];assert(!bad.open(1,128,48,512,one,primaries[0],src,fresh,err,false,true,&one_rank,&wrong));checks++;
available=512ull<<20;assert(!bad.open(1,128,48,512,one,primaries[0],src,fresh,err,false,true,&one_rank,&by_layer));available=16ull<<30;checks++;
// Each partial staging allocation failure releases all earlier resources and leaves claims untouched.
for(int fail=0;fail<=9;fail++) {
int a=allocations,b=streams;if(fail)failalloc=fail;else failstream=1;
assert(!bad.open(1,128,48,512,one,primaries[0],src,fresh,err,false,true,&one_rank,&by_layer));
assert(allocations==a && streams==b && current==0);
assert(std::none_of(fresh.begin(),fresh.end(),[](uint8_t v){return v!=0;}));checks+=3;
}
// Final memory reserve rejection and a host allocation exception have the same rollback contract.
for(int mode=0;mode<2;mode++) {
int a=allocations,b=streams;memcalls=0;if(mode)throwmem=2;else failmem=2;
assert(!bad.open(1,128,48,512,one,primaries[0],src,fresh,err,false,true,&one_rank,&by_layer));
assert(allocations==a && streams==b && current==0);
assert(std::none_of(fresh.begin(),fresh.end(),[](uint8_t v){return v!=0;}));
throwmem=failmem=0;checks+=3;
}
assert(!bad.open(1,128,INT64_MAX,INT64_MAX,one,primaries[0],src,fresh,err));checks++;
// Successful reuse after failure, and close while a window owns its staging buffers.
assert(bad.open(1,128,48,512,one,primaries[0],src,fresh,err,false,true,&one_rank,&by_layer));
assert(bad.begin(47,x,ids,8,10,kind,res,err));int before_sync=syncs;bad.close();assert(syncs>before_sync);

// Upstream auto-size boolean retains its original position and non-EP behavior.
RemoteExperts legacy;ExpertCache empty;std::vector<uint8_t> legacy_claimed(48*512);
std::vector<std::pair<int32_t,int32_t>> legacy_rank={{0,1},{0,2},{0,3}};
available=(512ull<<20)+512;assert(legacy.open(1,3,48,512,legacy_rank,empty,src,legacy_claimed,err,true));
assert(legacy.resident()==2);legacy.close();available=16ull<<30;checks+=2;
// Actual upstream optimized-decode branch and launch-mode snapshot; no rank rows published.
RemoteExpertOpt opt;RemoteExperts optimized;opt.attach(optimized);legacy_claimed.assign(48*512,0);
assert(optimized.open(1,3,48,512,legacy_rank,empty,src,legacy_claimed,err));assert(optimized.optimized_decode());
for(int i=0;i<80;i++){ids[i]=1;kind[i]=-1;}float weights[80]={};
opt.begin(weights,0,4);std::fill(out,out+80*32,-77);
assert(optimized.begin(0,x,ids,4,10,kind,nullptr,err));opt.end();
assert(optimized.finish(out,err));assert(reduced_calls==1&&accumulated_calls==1);
assert(std::all_of(out,out+80*32,[](float v){return v==-77;}));
assert(optimized.returned_bytes()==4*32*sizeof(float));checks+=7;
// Failed pre-reduction must drain and never accumulate stale helper results.
opt.begin(weights,0,4);reduce_fail=1;assert(!optimized.begin(0,x,ids,4,10,kind,nullptr,err));
assert(!optimized.finish(out,err));assert(accumulated_calls==1);checks+=3;
assert(optimized.begin(0,x,ids,4,10,kind,nullptr,err));assert(optimized.finish(nullptr,err));
assert(accumulated_calls==1);opt.end();checks+=3;
// Inactive opt keeps the original compact rank-row route.
assert(optimized.begin(0,x,ids,4,10,kind,nullptr,err));assert(optimized.finish(out,err));
assert(out[0]>0&&accumulated_calls==1);optimized.close();checks+=3;
// EP rejects both auto-size truncation and an attached pre-reducer before claims publish.
legacy_claimed.assign(48*512,0);RemoteExperts rejected;RemoteExpertOpt incompatible;incompatible.attach(rejected);
assert(!rejected.open(1,128,48,512,reset_rank,primary,src,legacy_claimed,err,false,true));
assert(std::none_of(legacy_claimed.begin(),legacy_claimed.end(),[](uint8_t v){return v!=0;}));
RemoteExperts truncated;assert(!truncated.open(1,128,48,512,reset_rank,primary,src,legacy_claimed,err,true,true));checks+=3;

for(auto& r:rs)r.close();for(auto& h:helpers)h.close();reset.close();
assert(allocations==0 && streams==0);checks+=4;
printf("PASS %d actual RemoteExperts host checks\\n",checks);}
'''
f=P/'actual_remote_host.cpp';f.write_text(stub+opt+policy+h+c+'\nnamespace strata::core {\nRemoteExpertOpt::~RemoteExpertOpt() {}\nvoid RemoteExpertOpt::attach(RemoteExperts& r) {r.remote_opt_=this;}\nsize_t RemoteExpertOpt::metadata_bytes(){return 64;}\nvoid RemoteExpertOpt::begin(const float* w,int b,int n){weights_=w;token_begin_=b;tokens_=n;}\nvoid RemoteExpertOpt::prepare(const RemoteExperts&,void* m)const{memset(m,0,64);}\nbool RemoteExpertOpt::reduce(RemoteExperts& r,const void*,std::string&){++reduced_calls;if(reduce_fail){--reduce_fail;return false;}memcpy(r.h_out_,r.d_out_,tokens_*32*sizeof(float));return true;}\nvoid RemoteExpertOpt::accumulate(const RemoteExperts&){++accumulated_calls;}\n}\n'+test)
results=[]
for label,flags in [('plain',[]),('asan-ubsan',['-fsanitize=address,undefined','-fno-omit-frame-pointer'])]:
 cmd=['clang++','-std=c++17','-O1',*flags,str(f),'-o',str(P/('host-'+label))]
 p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0,p.stderr
 p=subprocess.run([str(P/('host-'+label))],capture_output=True,text=True);assert p.returncode==0,p.stderr
 results.append({'mode':label,'command':cmd,'stdout':p.stdout})
(S.parent/'host-results.json').write_text(json.dumps(results,indent=2))
print(results)
