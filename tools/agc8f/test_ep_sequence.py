from pathlib import Path
import subprocess,json
S=Path(__file__).resolve().parents[2]
import tempfile
_tmp=tempfile.TemporaryDirectory(prefix='strata-ep-test-');P=Path(_tmp.name)
s=(S/'src/program/generate.cpp').read_text()
a=s.index('    // Bounded numerical test only:')
b=s.index('    const char* ep_dp',a)
c=s.index('            // Only validated requests consume sequence entries;')
d=s.index('            std::array<int64_t, 7> remote_before',c)
# Production driver placement is checked separately; extracted code retains its original return/continue behavior.
assert s.index('if (bad) { std::printf("ERR a token id is outside the vocabulary') < c < d
assert s.index('if (n + max_new + 8 > o.max_context)',s.index('while (next_line(line))')) < c
assert s.index('            cur = ids;',c)>d
assert s.count('EP_TEST_REQUEST index=')==1
prefix=r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cassert>
struct Helper { bool pending=false,active=true; int calls=0;
bool ep_set_active(bool v){++calls;if(pending)return false;active=v;return true;} };
int calls=0,accepted=0; Helper remote_experts[7];
int run(bool ep_l0,bool ep24,int count){
'''
suffix=r'''
for(int request=0;request<count;request++) {
'''+s[c:d]+r'''
++accepted;
}
return 0;
}
int main(int argc,char**argv){
assert(argc==5);int mode=atoi(argv[1]),count=atoi(argv[2]),pending=atoi(argv[3]),expect=atoi(argv[4]);
if(pending>=0)remote_experts[pending].pending=true;
int ret=run(mode==1,mode==2,count);
printf("RESULT ret=%d accepted=%d calls=",ret,accepted);
for(auto&h:remote_experts)printf("%d,",h.calls);puts("");
assert(ret==expect);
}
'''
f=P/'sequence_host.cpp';f.write_text(prefix+s[a:b]+suffix)
cases=[('absent',None,0,2,-1,0,2),('valid_l0','01',1,2,-1,0,2),('valid_ep24','0110',2,4,-1,0,4),
       ('exhausted','0',1,3,-1,0,1),('no_ep','01',0,1,-1,2,0),('empty','',1,1,-1,2,0),
       ('invalid','0x1',1,1,-1,2,0),('whitespace','01 ',1,1,-1,2,0),('max','0'*256,1,256,-1,0,256),
       ('too_long','0'*257,1,1,-1,2,0),('pending_first','0',1,1,0,2,0),('pending_last','0',2,1,6,2,0)]
import os
results=[]
for label,flags in [('plain',[]),('asan-ubsan',['-fsanitize=address,undefined','-fno-omit-frame-pointer'])]:
 cmd=['clang++','-std=c++17','-O1',*flags,str(f),'-o',str(P/('sequence-'+label))]
 subprocess.run(cmd,check=True,capture_output=True,text=True)
 for name,seq,mode,count,pending,expect,accepted in cases:
  env=dict(os.environ);env.pop('STRATA_EP_TEST_SEQUENCE',None)
  if seq is not None:env['STRATA_EP_TEST_SEQUENCE']=seq
  p=subprocess.run([str(P/('sequence-'+label)),str(mode),str(count),str(pending),str(expect)],env=env,capture_output=True,text=True)
  assert p.returncode==0,(name,p.stderr)
  assert f'accepted={accepted}' in p.stdout,(name,p.stdout)
  assert p.stderr.count('EP_TEST_REQUEST index=')==(accepted if seq else 0),(name,p.stderr)
  if name=='absent':assert 'calls=0,0,0,0,0,0,0,' in p.stdout
  if name=='valid_l0':assert 'calls=2,2,2,0,0,0,0,' in p.stdout
  if name=='valid_ep24':assert 'calls=4,4,4,4,4,4,4,' in p.stdout
  results.append(dict(mode=label,case=name,stdout=p.stdout,stderr=p.stderr))
(P/'sequence-results.json').write_text(json.dumps(results,indent=2))
print(f'PASS {len(results)} extracted production parser/boundary cases; static placement assertions pass')

# Compile the real compatibility guard: peer modes must fail before any EP setup.
ga=s.index('    const char* ep_dp =')
gb=s.index('    const bool remote_caches',ga)
guard=r'''
struct Options {bool serve=true,spec_split=false,no_pool=false;int adapt_swaps=0,kv_resident=0,peer_device=-1;std::string kv="fp16";int expert_cache_remote[3]={};};
int gate(bool ep_l0,bool ep24,Options o){
'''+s[ga:gb]+r'''
return 0;}
int main(){unsetenv("STRATA_VERIFY_DEVICE_PLAN");Options o;
assert(gate(false,true,o)==0);
for(int peer: {0,1,7}){o.peer_device=peer;assert(gate(false,true,o)==2);assert(gate(true,false,o)==2);assert(gate(false,false,o)==0);}
o.peer_device=-1;assert(gate(true,false,o)==0);puts("PASS EP peer compatibility guard");}
'''
gf=P/'guard.cpp';gf.write_text('#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <string>\n#include <cassert>\n'+guard)
subprocess.run(['clang++','-std=c++17',str(gf),'-o',str(P/'guard')],check=True)
subprocess.run([str(P/'guard')],check=True)

# Compile the production environment parser: explicit 0/1 only, default off.
pa=s.index('    // EP switches fail closed:')
pb=s.index('    // Bounded numerical test only:',pa)
parser='#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <cassert>\nint parse(){\n'+s[pa:pb]+r"""
return ep24 ? 24 : ep_l0 ? 1 : 0;
}
int main(){
unsetenv("STRATA_EP_L0");unsetenv("STRATA_EP_STAGE_TOP3");assert(parse()==0);
setenv("STRATA_EP_STAGE_TOP3","1",1);assert(parse()==24);
setenv("STRATA_EP_L0","1",1);assert(parse()==2);
setenv("STRATA_EP_STAGE_TOP3","0",1);assert(parse()==1);
setenv("STRATA_EP_L0","0",1);assert(parse()==0);
for(const char* key : {"STRATA_EP_L0","STRATA_EP_STAGE_TOP3"}) {
 for(const char* bad : {"", "true", "01", "2", "1 "}) {
  setenv(key,bad,1);assert(parse()==2);
 }
 setenv(key,"0",1);
}
puts("PASS strict EP environment parser");}
"""
pf=P/'parser.cpp';pf.write_text('#include <initializer_list>\n'+parser)
subprocess.run(['clang++','-std=c++17',str(pf),'-o',str(P/'parser')],check=True)
subprocess.run([str(P/'parser')],check=True)
