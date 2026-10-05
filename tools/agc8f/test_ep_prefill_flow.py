from pathlib import Path
import subprocess,json
import tempfile
S=Path(__file__).resolve().parents[2]
_tmp=tempfile.TemporaryDirectory(prefix='strata-ep-regression-');B=Path(_tmp.name)
t=(S/'src/prefill/prefill.cpp').read_text();a=t.index('bool Prefill::run_pipeline(');b=t.index('bool Prefill::run_impl(',a);c=t.index('bool Prefill::drain_pipeline(')
actual=t[a:b]+t[c:t.index('\n}  // namespace strata::prefill',c)]
stub=r'''
#include "strata/prefill/pipeline.hpp"
#include <algorithm>
#include <atomic>
#include <future>
#include <memory>
#include <cstring>
#include <cassert>
using Clock=std::chrono::steady_clock;double ms_since(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
using cudaError_t=int;constexpr int cudaSuccess=0;int cudaStreamSynchronize(void*){return 0;}const char*cudaGetErrorString(int){return "mock";}
namespace core{struct OnDevice{OnDevice(int){}};}
namespace detail=strata::prefill::detail;
struct Prefill{struct Impl{int device=0;void*cs=nullptr,*copy=nullptr;};std::unique_ptr<Impl>impl_=std::make_unique<Impl>();
struct Stats{double ms_total=0;}stats_;Prefill*next_=nullptr;int stage_lb_=0,hand_buf_=0;bool single_chunk_=false;
int capacity=2,fault=-1;const float*hand_in_=nullptr;std::future<bool>next_run_;std::string next_err_;
std::function<bool()>should_stop=[] {return false;};float slots[2]={};std::atomic<int>seen{0};bool audited=false;
int64_t chunk(){return capacity;}bool audit_state(int64_t,int64_t,std::string&){audited=true;return true;}
bool run_pipeline(const int64_t*,int64_t,int64_t,std::string&);bool run(const int64_t*,int64_t,int64_t,std::string&);bool drain_pipeline(std::string&);
bool run_impl(const int64_t*t,int64_t n,int64_t pos,std::string&e,detail::Pipeline*p=nullptr,size_t s=0,const detail::PipelineJob*input=nullptr){
 if(stage_lb_==fault){e="injected stage failure";return false;}if(p&&input)p->consumed(*input);
 for(int64_t i=0;i<n;i+=capacity){++seen;const int64_t count=std::min<int64_t>(capacity,n-i);
  if(next_){if(p){int slot=0;if(!p->acquire(s,slot)){e=p->error();return false;}slots[slot]=float(pos+i);
    if(!p->send(s+1,{t+i,count,pos+i,&slots[slot],s,slot})){e=p->error();return false;}}
   else{if(next_run_.valid()&&!next_run_.get()){e=next_err_;return false;}next_->single_chunk_=n<=capacity;
    next_run_=std::async(std::launch::async,[=,this]{return next_->run_impl(t+i,count,pos+i,next_err_);});}}
 }return true;
}
};
'''
test=r'''
int main(){int checks=0;int64_t tokens[8]={};for(int scenario=0;scenario<6;scenario++){
 Prefill stages[8];for(int i=0;i<8;i++){stages[i].stage_lb_=i;stages[i].impl_->device=i;if(i<7)stages[i].next_=stages+i+1;}
 setenv("STRATA_PREFILL_PIPELINE",scenario==0?"0":"1",1);setenv("STRATA_PREFILL_PIPELINE_AUDIT","1",1);
 if(scenario==2)stages[6].capacity=3;if(scenario==3)stages[4].fault=4;
 if(scenario==4)stages[7].next_=stages+2;if(scenario==5)setenv("STRATA_DBG_NAN","1",1);
 std::string err;const bool ok=stages[0].run(tokens,8,0,err);assert(ok==(scenario<=2));checks++;
 if(ok){for(auto&s:stages){assert(s.seen==4&&!s.next_run_.valid());checks++;}assert(stages[0].audited);checks++;}
 else{assert(!err.empty()&&!stages[0].audited);checks++;}
 unsetenv("STRATA_DBG_NAN");
}printf("PASS %d actual run_pipeline/request wrapper checks (6 scenarios)\n",checks);}
'''
f=B/'flow.cpp';f.write_text(stub+actual+test);results=[]
for name,flags in [('plain',[]),('asan-ubsan',['-fsanitize=address,undefined','-fno-omit-frame-pointer'])]:
 q=subprocess.run(['clang++','-std=c++17','-O1','-pthread','-I'+str(S/'include'),*flags,str(f),'-o',str(B/name)],capture_output=True,text=True);assert not q.returncode,q.stderr
 q=subprocess.run([str(B/name)],capture_output=True,text=True);assert not q.returncode,q.stderr
 results.append({'mode':name,'stdout':q.stdout,'stderr':q.stderr,'mocked':'run_impl compute body, CUDA streams/device; actual queue/request wrapper/drain'})
print(json.dumps(results,indent=2))
