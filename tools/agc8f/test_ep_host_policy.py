"""Host-only EP policy regression: extract current repo statements, no old tree needed.

Run from any cwd. --out retains generated sources, binaries and JSON in a new
output directory; without it, artifacts live in an automatically removed temp dir.
CUDA, graph execution and model mathematics are mocked and not validated here.
"""
from pathlib import Path
import argparse, ast, json, os, re, subprocess, sys, tempfile
S = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, help='new directory for retained test artifacts')
args = parser.parse_args()
_tmp = None
if args.out is None:
    _tmp = tempfile.TemporaryDirectory(prefix='strata-ep-host-policy-')
    B = Path(_tmp.name)
else:
    B = args.out.resolve()
    B.mkdir(parents=True, exist_ok=False)
verify = (S / 'src/core/verify.cpp').read_text()
header = (S / 'include/strata/core/verify.hpp').read_text()
gen = (S / 'src/program/generate.cpp').read_text()
def policy(text):
    a = text.index('    all_resident_ = false;', text.index('bool Verifier::init('))
    b = text.index('    if (all_resident_ || device_plan_)', a)
    return text[a:b]
setter = re.search(r'    void require_host_expert_dispatch\(bool on\) \{[^\n]+\}', header).group()
wire = re.search(r'        for \(int st = 0; st < n_stages; \+\+st\) stage_ver\(st\).require_host_expert_dispatch\(ep_l0 \|\| ep24\);', gen).group()
# Source topology: the real wiring precedes every serve-stage init, not merely EP_INIT.
serve = gen[gen.index('        SplitDrive split_drive;'):gen.index('        std::vector<int64_t> cur;', gen.index('        SplitDrive split_drive;'))]
assert serve.index(wire) < serve.index('ver_same.init(')
assert serve.index(wire) < serve.index('gs.ver.init(')
assert serve.index(wire) < serve.index('ver.init(')
assert 'bool host_expert_dispatch_ = false;' in header
assert verify.count('if (all_resident_ && !test_stall) {') == 1
condition = re.search(r'if \(all_resident_ && !test_stall\)', verify).group()
# Use actual production split callback and dispatch with the existing dependency stubs.
parsed = ast.parse((S / 'tools/agc8f/test_ep_dispatch.py').read_text())
values = {n.targets[0].id: ast.literal_eval(n.value) for n in parsed.body
          if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)}
stub = values['stub']
exp = (S / 'src/core/expert_source.cpp').read_text()
actual_dispatch = exp[exp.index('namespace {\n// the verify window'):exp.index('\nvoid expert_hit_run(', exp.index('namespace {\n// the verify window'))]
callbacks = gen[gen.index('void drive_pool_multi('):gen.index('/// Layer split across GPUs:', gen.index('void drive_pool_multi('))]
callbacks = callbacks.replace('strata::core::RemoteExperts*', 'strata::core::Remote*')
# Type name only adapted to the Remote dependency stub; callback statements remain verbatim.
class_stub = r'''
struct Drive{ExpertDispatch d;double cpu_ms=0;int calls=0;FILE*routing=nullptr;};
using Clock=std::chrono::steady_clock;
struct VerifyHits{const int32_t*h_res=nullptr,*d_res=nullptr;const uint8_t*cache_base=nullptr;};
struct Geometry{int n_expert=512,n_embd=32;};
struct Verifier{bool host_expert_dispatch_=false,all_resident_=false,device_plan_=false;
int lb_=0,le_=6;Geometry g;VerifyHits hits;
SETTER
void init(){POLICY}
// Exercise Verifier::run's exact host bypass predicate, then its exact pool invocation.
void serve(void(*pool)(void*,const float*,const int32_t*,int64_t,int64_t,float*,int64_t),void*user,
           const float*h_x_,const int32_t*h_ids_,float*h_ymiss_,int n,int l){
 bool test_stall=false;int tb=0;struct{int k=10;}ss;
 CONDITION{ return; } else {
 CALL
 }
}
};
'''
run = verify[verify.index('bool Verifier::run('):verify.index('void Verifier::set_plan_slot(')]
call = re.search(r'        if \(pool != nullptr\)\n            pool\(user,[\s\S]*?h_ymiss_ \+ \(size_t\) tb \* ss.k \* g.n_embd, l\);', run).group()
new_class = class_stub.replace('SETTER', setter).replace('POLICY', policy(verify)).replace('CONDITION', condition).replace('CALL', call)
# Construct the pre-repair policy locally, removing exactly the two EP guards.
# This is a targeted mutation negative control, not a historical-source equality claim.
current_policy = policy(verify)
assert current_policy.count('!host_expert_dispatch_ && ') == 2
baseline_policy = current_policy.replace('!host_expert_dispatch_ && ', '')
old_class = class_stub.replace('SETTER', setter).replace('POLICY', baseline_policy).replace('CONDITION', condition).replace('CALL', call)
tests = r'''
}
int main(){using namespace strata::core;int checks=0;
int32_t res[48*512];std::fill(res,res+48*512,0);uint8_t cache[512*8]={};
for(int mode=0;mode<3;mode++)for(int ar=-1;ar<=1;ar++)for(int dp=0;dp<=1;dp++)for(int missing=0;missing<=1;missing++){
 if(ar<0)unsetenv("STRATA_VERIFY_ALL_RESIDENT");else setenv("STRATA_VERIFY_ALL_RESIDENT",ar?"1":"0",1);
 setenv("STRATA_VERIFY_DEVICE_PLAN",dp?"1":"0",1);
 bool ep_l0=mode==1,ep24=mode==2;int n_stages=8;Verifier ver[8];
 auto stage_ver=[&](int st)->Verifier&{return ver[st];};
 WIRE
 res[0]=missing?-1:0;
 for(int st=0;st<8;st++){
  auto&v=ver[st];v.lb_=st*6;v.le_=(st+1)*6;v.hits={res,res,cache};v.init();
  bool expected_ar=mode==0&&ar!=0&&(st!=0||!missing);
  assert(v.all_resident_==expected_ar);assert(v.device_plan_==(mode==0&&!expected_ar&&dp));checks+=2;
 }
}
unsetenv("STRATA_VERIFY_ALL_RESIDENT");unsetenv("STRATA_VERIFY_DEVICE_PLAN");res[0]=0;
Source src;Pool pool;Remote rem[3];for(int r=0;r<3;r++)rem[r].r=r+1;
int32_t ids[80];for(int i=0;i<80;i++)ids[i]=(i*7)%32;float x[8*H]={},out[80*H];
for(int mode=0;mode<3;mode++){
 Drive d;SplitDrive split;split.base=&d;split.n=8;split.ep24=mode==2;
 d.d.src=&src;d.d.pool=&pool;d.d.host_res=res;
 GpuPlanSink plans[8];Verifier ver[8];bool ep_l0=mode==1,ep24=mode==2;int n_stages=8;
 auto stage_ver=[&](int st)->Verifier&{return ver[st];};
 WIRE
 for(int st=0;st<8;st++){
  split.end[st]=(st+1)*6;split.plan[st]=plans+st;split.cache_base[st]=cache;
  for(int r=0;r<3;r++)split.ep_remote[st][r]=rem+r;
  ver[st].hits={res,res,cache};ver[st].init();
 }
 if(ep_l0){d.d.remote_count=3;for(int r=0;r<3;r++)d.d.remote[r]=rem+r;}
 for(int T: {3,4,6,8}){
  std::fill(out,out+80*H,-5);for(auto&r:rem)r.begins=r.drains=r.publishes=0;
  ver[0].serve(&drive_pool_split,&split,x,ids,out,T,0);
  assert(!d.d.failed);checks++;
  if(mode==0){assert(d.calls==0);for(auto&r:rem)assert(r.begins==0);checks+=4;}
  else{
   for(int r=0;r<3;r++){
    assert(rem[r].begins==1&&rem[r].drains==1&&!rem[r].pending);checks++;
    for(int i=0;i<T*10;i++)if(ids[i]%4==r+1){assert(out[i*H]==(T>=4?float((r+1)*1000+i):0));checks++;}
   }
  }
 }
}
printf("PASS %d actual policy/callback/dispatch checks; 36 policy cases x 8 stages, synthetic EP T3 fallback/T4,T6,T8 dispatch\n",checks);
}
'''.replace('WIRE',wire)
results=[]
negative_tests = tests.replace('assert(v.all_resident_==expected_ar);',
    'if(v.all_resident_!=expected_ar){std::fprintf(stderr,"HOST_POLICY_REGRESSION_DETECTED\\n");return 23;}')
for label,flags,cls,expect in [('plain',[],new_class,0),('asan-ubsan',['-fsanitize=address,undefined','-fno-omit-frame-pointer'],new_class,0),('baseline',[],old_class,23)]:
    f=B/(label+'.cpp');f.write_text(stub+actual_dispatch+cls+callbacks+(negative_tests if expect else tests))
    q=subprocess.run(['clang++','-std=c++17','-O1',*flags,str(f),'-o',str(B/label)],capture_output=True,text=True)
    assert q.returncode==0,q.stderr
    env={**os.environ,"STRATA_DEC_BATCH":"0","ASAN_OPTIONS":("detect_leaks=1" if sys.platform.startswith("linux") else "detect_leaks=0")+":halt_on_error=1","UBSAN_OPTIONS":"halt_on_error=1:print_stacktrace=1"}
    q=subprocess.run([str(B/label)],capture_output=True,text=True,env=env)
    assert q.returncode==expect,(q.returncode,q.stdout,q.stderr)
    if expect:
        assert q.stderr == 'HOST_POLICY_REGRESSION_DETECTED\n',q.stderr
    else:
        assert q.stdout.startswith('PASS 942 actual policy/callback/dispatch checks;'),q.stdout
        assert not q.stderr,q.stderr
    results.append({'mode':label,'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'expected_failure':bool(expect)})
report={'results':results,'source_wiring_before_all_stage_init':True,
        'negative_control':'local mutation removes exactly two host_expert_dispatch_ policy guards; no old source tree required',
        'limits':'CUDA calls/graph not executed; real policy, bypass predicate, pool call, split callback and dispatcher extracted'}
(B/'policy-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
