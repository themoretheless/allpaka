from pathlib import Path
import subprocess,shutil
root=Path('/tmp/allpaka-profile-20261025');metal=Path('crates/allpaka-backend/src/gpu/metal.rs');kern=Path('crates/allpaka-backend/src/gpu/metal_kernels.rs')
a,b=metal.read_text(),kern.read_text()
old='''        // Only the shared expert's down projection consumes the swiglu
        // written above. Without a shared expert, the combine barrier below
        // already orders all expert down outputs; draining the same arena
        // here would add a second barrier without an intervening dispatch.
        if req.shared.is_some() && !no_barrier() && !no_stage_barrier() {'''
assert a.count(old)==1
basea=a.replace(old,'        if !no_barrier() && !no_stage_barrier() {');baseb=(root/'metal_kernels-before.rs').read_text()
(root/'metal-after.rs').write_text(a);(root/'kernels-after.rs').write_text(b)
try:
 metal.write_text(basea);kern.write_text(baseb)
 with (root/'build-forced-before.log').open('w') as f:subprocess.run(['cargo','build','--release','-p','allpaka-model','--example','shared_activation_bench','--offline'],stdout=f,stderr=subprocess.STDOUT,check=True)
 shutil.copy2('target/release/examples/shared_activation_bench',root/'forced-before')
finally:
 assert metal.read_text()==basea and kern.read_text()==baseb,'concurrent source edit; restore only own hunk manually'
 metal.write_text(a);kern.write_text(b)
with (root/'build-forced-after.log').open('w') as f:subprocess.run(['cargo','build','--release','-p','allpaka-model','--example','shared_activation_bench','--offline'],stdout=f,stderr=subprocess.STDOUT,check=True)
shutil.copy2('target/release/examples/shared_activation_bench',root/'forced-after')
