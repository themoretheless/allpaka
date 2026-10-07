//! Server-side online evaluation rule admission and deterministic trace selection.
//! Execution and durable rule lifecycle are separate from this definition.
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Rule {
    pub id: String,
    pub project_id: String,
    pub evaluator_id: String,
    pub evaluator_version: u64,
    pub sample_rate: f64,
    pub enabled: bool,
}
fn technical(value: &str) -> bool {
    !value.is_empty() && value.len() <= 100
        && value.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}
impl Rule {
    pub fn validate(&self) -> Result<()> {
        if !technical(&self.id) || !technical(&self.project_id) || !technical(&self.evaluator_id)
            || self.evaluator_version == 0 || !self.sample_rate.is_finite()
            || !(0.0..=1.0).contains(&self.sample_rate) {
            bail!("Invalid online evaluation rule");
        }
        Ok(())
    }
    pub fn selects(&self, project: &str, trace_id: &str) -> Result<bool> {
        self.validate()?;
        if !technical(project) || !technical(trace_id) {bail!("Invalid evaluation trace identity");}
        if !self.enabled || project != self.project_id || self.sample_rate == 0.0 {return Ok(false);}
        if self.sample_rate == 1.0 {return Ok(true);}
        // Independent of scan order, execution timing and process restart.
        // Evaluator revisions intentionally receive distinct samples.
        let key=serde_json::to_vec(&("allpaka-online-evaluation-v1",&self.project_id,&self.id,
            &self.evaluator_id,self.evaluator_version,trace_id))?;
        let digest=Sha256::digest(key);
        let bucket=u64::from_be_bytes(digest[..8].try_into().unwrap()) >> 11;
        Ok((bucket as f64)/((1u64<<53) as f64) < self.sample_rate)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Snapshot {
    pub kind: String,
    pub schema_version: u32,
    pub rule_sha256: String,
    pub rule: Rule,
}
impl Snapshot {
    fn new(rule: Rule) -> Result<Self> {
        rule.validate()?;
        let hash=Sha256::digest(serde_json::to_vec(&rule)?).iter().map(|b|format!("{b:02x}")).collect();
        Ok(Self{kind:"online_evaluation_rule".into(),schema_version:1,rule_sha256:hash,rule})
    }
    fn validate(&self)->Result<()> {
        let expected=Self::new(self.rule.clone())?;
        if self.kind!=expected.kind||self.schema_version!=1||self.rule_sha256!=expected.rule_sha256 {bail!("Online evaluation rule integrity mismatch");}
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Binding {kind:String,project_id:String,rule_id:String,rule_sha256:String,version:u64,active:bool,binding_sha256:String}
impl Binding {
    fn hash(&self)->Result<String>{Ok(Sha256::digest(serde_json::to_vec(&(&self.kind,&self.project_id,&self.rule_id,&self.rule_sha256,self.version,self.active))?).iter().map(|b|format!("{b:02x}")).collect())}
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BindInput {rule_sha256:String,base_version:u64,active:bool}
static WRITES:std::sync::Mutex<()>=std::sync::Mutex::new(());
struct Store {directory:std::path::PathBuf}
impl Store {
    fn open(data:&std::path::Path)->Result<Self>{
        let path=data.join("observability/online-rules");std::fs::create_dir_all(&path)?;
        let directory=path.canonicalize()?;if !directory.starts_with(data){bail!("Online rule storage escapes data directory");}
        Ok(Self{directory})
    }
    fn path(&self,hash:&str)->Result<std::path::PathBuf>{
        if hash.len()!=64||!hash.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b)){bail!("Invalid rule hash");}
        Ok(self.directory.join(format!("{hash}.json")))
    }
    fn get(&self,hash:&str)->Result<Snapshot>{
        use std::io::Read;
        let path=self.path(hash)?;
        if !std::fs::symlink_metadata(&path)?.file_type().is_file(){bail!("Invalid rule file");}
        let mut bytes=Vec::new();std::fs::File::open(path)?.take(8193).read_to_end(&mut bytes)?;
        if bytes.len()>8192 {bail!("Rule exceeds read limit");}
        let value=crate::strict_json::parse(std::str::from_utf8(&bytes)?)?;
        let snapshot:Snapshot=serde_json::from_value(value)?;snapshot.validate()?;
        if snapshot.rule_sha256!=hash {bail!("Rule filename mismatch");}Ok(snapshot)
    }
    fn binding_directory(&self,project:&str,rule:&str)->Result<std::path::PathBuf>{
        if !technical(project)||!technical(rule){bail!("Invalid binding identity");}
        let key=Sha256::digest(serde_json::to_vec(&(project,rule))?).iter().map(|b|format!("{b:02x}")).collect::<String>();
        let parent=self.directory.join("bindings");if parent.exists()&&!parent.canonicalize()?.starts_with(&self.directory){bail!("Binding directory escapes rule storage");}
        let path=parent.join(key);if !path.exists(){return Ok(path);}
        let path=path.canonicalize()?;if !path.starts_with(&self.directory){bail!("Binding directory escapes rule storage");}Ok(path)
    }
    fn binding(&self,project:&str,rule:&str)->Result<Option<Binding>>{
        use std::io::Read;
        let directory=self.binding_directory(project,rule)?;if !directory.exists(){return Ok(None);}let mut versions=Vec::new();
        for entry in std::fs::read_dir(&directory)?{let entry=entry?;let path=entry.path();if path.extension().and_then(|s|s.to_str())!=Some("json"){continue;}
            let name=path.file_stem().and_then(|s|s.to_str()).ok_or_else(||anyhow::anyhow!("Invalid binding filename"))?;
            let version=name.parse::<u64>()?;if version==0||version>1000||name!=version.to_string()||!entry.file_type()?.is_file(){bail!("Invalid binding revision");}
            versions.push(version);if versions.len()>1000{bail!("Binding history exceeds limit");}}
        versions.sort_unstable();if versions.iter().enumerate().any(|(index,version)|*version!=index as u64+1){bail!("Incomplete binding history");}
        let Some(version)=versions.last()else{return Ok(None)};
        let mut bytes=Vec::new();std::fs::File::open(directory.join(format!("{version}.json")))?.take(8193).read_to_end(&mut bytes)?;
        if bytes.len()>8192{bail!("Binding exceeds read limit");}
        let binding:Binding=serde_json::from_value(crate::strict_json::parse(std::str::from_utf8(&bytes)?)?)?;
        if binding.kind!="online_evaluation_binding"||binding.project_id!=project||binding.rule_id!=rule||binding.version!=*version||binding.binding_sha256!=binding.hash()?{bail!("Binding integrity mismatch");}
        let snapshot=self.get(&binding.rule_sha256)?;
        if snapshot.rule.project_id!=project||snapshot.rule.id!=rule||binding.active&&!snapshot.rule.enabled{bail!("Binding rule mismatch");}Ok(Some(binding))
    }
    fn bind(&self,input:BindInput)->Result<Binding>{
        let _lock=WRITES.lock().map_err(|_|anyhow::anyhow!("Rule storage lock unavailable"))?;
        let snapshot=self.get(&input.rule_sha256)?;
        let current=self.binding(&snapshot.rule.project_id,&snapshot.rule.id)?;
        if input.base_version!=current.as_ref().map(|b|b.version).unwrap_or(0)||input.base_version>=1000{bail!("Stale or exhausted binding revision");}
        if input.active&&!snapshot.rule.enabled{bail!("Cannot activate disabled rule snapshot");}
        let mut binding=Binding{kind:"online_evaluation_binding".into(),project_id:snapshot.rule.project_id,rule_id:snapshot.rule.id,rule_sha256:input.rule_sha256,version:input.base_version+1,active:input.active,binding_sha256:String::new()};
        binding.binding_sha256=binding.hash()?;
        let directory=self.binding_directory(&binding.project_id,&binding.rule_id)?;std::fs::create_dir_all(&directory)?;let directory=directory.canonicalize()?;if !directory.starts_with(&self.directory){bail!("Binding directory escapes rule storage");}
        crate::evaluation::commit_new(&directory.join(format!("{}.json",binding.version)),&binding)?;Ok(binding)
    }
    fn process_job(&self,path:&std::path::Path,data:&std::path::Path,automatic:bool,inspect_only:bool)->Result<Option<bool>>{
        use std::io::Read;
        if !std::fs::symlink_metadata(path)?.file_type().is_file(){bail!("Invalid queue job file");}
        let mut bytes=Vec::new();std::fs::File::open(path)?.take(8193).read_to_end(&mut bytes)?;if bytes.len()>8192{bail!("Job exceeds read limit");}
        let job=crate::strict_json::parse(std::str::from_utf8(&bytes)?)?;
        let project=job["project_id"].as_str().ok_or_else(||anyhow::anyhow!("Missing job project"))?;let trace=job["trace_id"].as_str().ok_or_else(||anyhow::anyhow!("Missing job trace"))?;
        let key=job["selection_sha256"].as_str().ok_or_else(||anyhow::anyhow!("Missing job selection"))?;
        if job.as_object().map(|v|v.len())!=Some(8)||job["kind"]!="online_evaluation_job"||job["schema_version"]!=1||job["status"]!="pending"||job["provider_calls"]!=0||!technical(project)||!technical(trace)||key.len()!=64||!key.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b))||path.file_stem().and_then(|s|s.to_str())!=Some(key){bail!("Invalid pinned queue job");}
        let job_hash=Sha256::digest(serde_json::to_vec(&job)?).iter().map(|b|format!("{b:02x}")).collect::<String>();
        let directory=self.directory.join("job-results");if !inspect_only{std::fs::create_dir_all(&directory)?;}if directory.exists()&&!directory.canonicalize()?.starts_with(&self.directory){bail!("Job results escape storage");}let result_path=directory.join(format!("{key}.json"));
        if result_path.try_exists()?{if !std::fs::symlink_metadata(&result_path)?.file_type().is_file(){bail!("Invalid job result file");}let mut bytes=Vec::new();std::fs::File::open(result_path)?.take(8193).read_to_end(&mut bytes)?;if bytes.len()>8192{bail!("Job result exceeds limit");}let mut result=crate::strict_json::parse(std::str::from_utf8(&bytes)?)?;let hash=result.as_object_mut().ok_or_else(||anyhow::anyhow!("Invalid job result"))?.remove("result_sha256").ok_or_else(||anyhow::anyhow!("Missing result digest"))?;
            let expected=Sha256::digest(serde_json::to_vec(&result)?).iter().map(|b|format!("{b:02x}")).collect::<String>();if result.as_object().map(|v|v.len())!=Some(9)||hash!=expected||result["kind"]!="online_evaluation_job_result"||result["job_sha256"]!=job_hash||result["selection_sha256"]!=key||!matches!(result["status"].as_str(),Some("completed"|"failed")){bail!("Job result integrity mismatch");}if result["schema_version"]!=1||result["provider_calls"]!=0||!result["automatic_execution"].is_boolean(){bail!("Invalid completed job result");}
            if (result["status"]=="completed" && !result["error_code"].is_null()) || (result["status"]=="failed" && !matches!(result["error_code"].as_str(),Some("evaluator_failed"|"assessment_rejected"))){bail!("Invalid job outcome");}
            if let Some(hash)=result["assessment_sha256"].as_str(){let assessment_path=self.directory.join("assessments").join(format!("{key}.json"));if !std::fs::symlink_metadata(&assessment_path)?.file_type().is_file(){bail!("Invalid retained assessment");}let mut bytes=Vec::new();std::fs::File::open(assessment_path)?.take(1024*1024+1).read_to_end(&mut bytes)?;if bytes.len()>1024*1024{bail!("Retained assessment exceeds limit");}let assessment=crate::strict_json::parse(std::str::from_utf8(&bytes)?)?;let actual=Sha256::digest(serde_json::to_vec(&assessment)?).iter().map(|b|format!("{b:02x}")).collect::<String>();if hash!=actual{bail!("Retained assessment evidence mismatch");}}
            else if result["status"]=="completed"{bail!("Completed job lacks assessment evidence");}
            return Ok(None);}
        if inspect_only{return Ok(Some(false));}
        let attempted=(||->Result<serde_json::Value>{let (hash,health)=crate::observability::Store::new(data)?.completed_trace_health(trace,project)?;if job["trace_sha256"]!=hash{bail!("Job trace evidence changed");}
            let archive=self.read_selection_archive(project,trace)?;if archive["selection"]["selection_sha256"]!=key||archive["trace_evidence_pinned"]!=true{bail!("Job selection evidence changed");}
            self.assess_trace_origin(project,trace,&hash,health,automatic)
        })();
        let (success,error,assessment_hash)=match attempted{Ok(assessment)=>{let success=assessment["assessments"].as_array().unwrap().iter().all(|item|item["status"]=="completed");let hash=Sha256::digest(serde_json::to_vec(&assessment)?).iter().map(|b|format!("{b:02x}")).collect::<String>();(success,if success{serde_json::Value::Null}else{serde_json::json!("evaluator_failed")},serde_json::json!(hash))},Err(_)=>(false,serde_json::json!("assessment_rejected"),serde_json::Value::Null)};
        let mut result=serde_json::json!({"kind":"online_evaluation_job_result","schema_version":1,"selection_sha256":key,"job_sha256":job_hash,"status":if success{"completed"}else{"failed"},"error_code":error,"assessment_sha256":assessment_hash,"provider_calls":0,"automatic_execution":automatic});
        result["result_sha256"]=serde_json::json!(Sha256::digest(serde_json::to_vec(&result)?).iter().map(|b|format!("{b:02x}")).collect::<String>());
        crate::evaluation::commit_new(&result_path,&result)?;Ok(Some(success))
    }
    fn enqueue_trace(&self,project:&str,trace:&str,trace_hash:&str)->Result<Option<serde_json::Value>>{
        let mut active=false;
        for offset in (0..1000).step_by(100){let catalog=self.list(project,offset,100)?;
            for row in catalog["rules"].as_array().unwrap(){let id=row["rule"]["id"].as_str().unwrap();if self.binding(project,id)?.is_some_and(|binding|binding.active){active=true;break;}}
            if active||catalog["has_more"]==false{break;}}
        if !active{return Ok(None);}
        let selection=self.select_trace(project,trace,trace_hash)?;
        if !selection["decisions"].as_array().unwrap().iter().any(|decision|decision["selected"]==true&&serde_json::from_value::<Binding>(decision["binding"].clone()).ok().and_then(|binding|self.get(&binding.rule_sha256).ok()).is_some_and(|snapshot|!snapshot.rule.evaluator_id.starts_with("model_quality."))){return Ok(None);}
        let packet=serde_json::json!({"kind":"online_evaluation_job","schema_version":1,"project_id":project,"trace_id":trace,"trace_sha256":trace_hash,"selection_sha256":selection["selection_sha256"],"status":"pending","provider_calls":0});
        let _lock=WRITES.lock().map_err(|_|anyhow::anyhow!("Rule storage lock unavailable"))?;
        let directory=self.directory.join("jobs");std::fs::create_dir_all(&directory)?;let directory=directory.canonicalize()?;if !directory.starts_with(&self.directory){bail!("Job directory escapes storage");}
        let path=directory.join(format!("{}.json",selection["selection_sha256"].as_str().unwrap()));
        if path.try_exists()?{use std::io::Read;if !std::fs::symlink_metadata(&path)?.file_type().is_file(){bail!("Invalid job file");}let mut bytes=Vec::new();std::fs::File::open(path)?.take(8193).read_to_end(&mut bytes)?;if bytes.len()>8192{bail!("Job exceeds read limit");}let stored=crate::strict_json::parse(std::str::from_utf8(&bytes)?)?;if stored!=packet{bail!("Job identity mismatch");}return Ok(Some(stored));}
        let mut count=0;for entry in std::fs::read_dir(&directory)?{let entry=entry?;if entry.path().extension().and_then(|s|s.to_str())!=Some("json"){continue;}count+=1;if count>=1000||!entry.file_type()?.is_file(){bail!("Online job capacity reached");}}
        crate::evaluation::commit_new(&path,&packet)?;Ok(Some(packet))
    }
    fn assess_trace(&self,project:&str,trace:&str,trace_hash:&str,health:f64)->Result<serde_json::Value>{
        self.assess_trace_origin(project,trace,trace_hash,health,false)
    }
    fn assess_trace_origin(&self,project:&str,trace:&str,trace_hash:&str,health:f64,automatic:bool)->Result<serde_json::Value>{
        use std::io::Read;
        if !health.is_finite()||!(0.0..=1.0).contains(&health){bail!("Invalid native trace health");}
        let selection=self.select_trace(project,trace,trace_hash)?;
        let mut assessments=Vec::new();
        for decision in selection["decisions"].as_array().unwrap(){if decision["selected"]!=true{continue;}
            let binding:Binding=serde_json::from_value(decision["binding"].clone())?;let snapshot=self.get(&binding.rule_sha256)?;
            if snapshot.rule.evaluator_id.starts_with("model_quality."){continue;}
            let supported=snapshot.rule.evaluator_id=="trace_health"&&snapshot.rule.evaluator_version==1;
            assessments.push(serde_json::json!({"rule_id":binding.rule_id,"rule_sha256":binding.rule_sha256,"binding_sha256":binding.binding_sha256,"binding_version":binding.version,"evaluator_id":snapshot.rule.evaluator_id,"evaluator_version":snapshot.rule.evaluator_version,"status":if supported{"completed"}else{"failed"},"scores":if supported{serde_json::json!({"span_success_rate":health})}else{serde_json::json!({})},"error_code":if supported{serde_json::Value::Null}else{serde_json::json!("evaluator_unavailable")}}));
        }
        let mut packet=serde_json::json!({"kind":"online_evaluation_assessments","schema_version":1,"project_id":project,"trace_id":trace,"trace_sha256":trace_hash,"selection_sha256":selection["selection_sha256"],"assessments":assessments,"assessment_source":"native_trace_metadata","provider_calls":0,"automatic_execution":automatic});
        if serde_json::to_vec(&packet)?.len()>1024*1024{bail!("Assessment exceeds write limit");}
        let _lock=WRITES.lock().map_err(|_|anyhow::anyhow!("Rule storage lock unavailable"))?;
        let directory=self.directory.join("assessments");std::fs::create_dir_all(&directory)?;let directory=directory.canonicalize()?;if !directory.starts_with(&self.directory){bail!("Assessment directory escapes storage");}
        let key=selection["selection_sha256"].as_str().unwrap();let path=directory.join(format!("{key}.json"));
        if path.try_exists()?{if !std::fs::symlink_metadata(&path)?.file_type().is_file(){bail!("Invalid assessment file");}
            let mut bytes=Vec::new();std::fs::File::open(path)?.take(1024*1024+1).read_to_end(&mut bytes)?;if bytes.len()>1024*1024{bail!("Assessment exceeds read limit");}
            let stored=crate::strict_json::parse(std::str::from_utf8(&bytes)?)?;if !stored["automatic_execution"].is_boolean(){bail!("Invalid assessment execution origin");}packet["automatic_execution"]=stored["automatic_execution"].clone();if stored!=packet{bail!("Assessment evidence mismatch");}return Ok(stored);}
        let mut count=0;for entry in std::fs::read_dir(&directory)?{let entry=entry?;if entry.path().extension().and_then(|s|s.to_str())!=Some("json"){continue;}count+=1;if count>=1000||!entry.file_type()?.is_file(){bail!("Assessment storage capacity reached");}}
        crate::evaluation::commit_new(&path,&packet)?;Ok(packet)
    }
    fn read_selection_archive(&self,project:&str,trace:&str)->Result<serde_json::Value>{
        use std::io::Read;
        if !technical(project)||!technical(trace){bail!("Invalid selection identity");}
        let key=Sha256::digest(serde_json::to_vec(&(project,trace))?).iter().map(|b|format!("{b:02x}")).collect::<String>();
        let path=self.directory.join("selections").join(format!("{key}.json"));
        if !std::fs::symlink_metadata(&path)?.file_type().is_file()||!path.canonicalize()?.starts_with(&self.directory){bail!("Invalid selection archive path");}
        let mut bytes=Vec::new();std::fs::File::open(path)?.take(1024*1024+1).read_to_end(&mut bytes)?;if bytes.len()>1024*1024{bail!("Selection archive exceeds read limit");}
        let packet=crate::strict_json::parse(std::str::from_utf8(&bytes)?)?;
        let pinned=if packet["schema_version"]==2{
            let hash=packet["trace_sha256"].as_str().ok_or_else(||anyhow::anyhow!("Missing trace hash"))?;
            if hash.len()!=64||!hash.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b)){bail!("Invalid archived trace hash");}
            self.validate_selection(&packet,project,trace,hash)?;true
        }else if packet["schema_version"]==1{
            if packet.as_object().map(|o|o.len())!=Some(8){bail!("Invalid legacy selection shape");}
            let decisions=packet["decisions"].as_array().ok_or_else(||anyhow::anyhow!("Missing legacy decisions"))?;
            let digest=Sha256::digest(serde_json::to_vec(&(project,trace,decisions))?).iter().map(|b|format!("{b:02x}")).collect::<String>();if packet["selection_sha256"]!=digest{bail!("Legacy selection integrity mismatch");}
            // Validate common decision semantics without inventing evidence in the retained file.
            let placeholder="0".repeat(64);let mut checked=packet.clone();checked["schema_version"]=serde_json::json!(2);checked["trace_sha256"]=serde_json::json!(placeholder);
            checked["selection_sha256"]=serde_json::json!(Sha256::digest(serde_json::to_vec(&(project,trace,&placeholder,decisions))?).iter().map(|b|format!("{b:02x}")).collect::<String>());
            self.validate_selection(&checked,project,trace,&placeholder)?;false
        }else{bail!("Unsupported archived selection schema");};
        Ok(serde_json::json!({"kind":"online_evaluation_selection_archive","selection":packet,"trace_evidence_pinned":pinned,"execution_eligible":false,"provider_calls":0}))
    }
    fn select_trace(&self,project:&str,trace:&str,trace_hash:&str)->Result<serde_json::Value>{
        use std::io::Read;
        if !technical(project)||!technical(trace)||trace_hash.len()!=64||!trace_hash.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b)){bail!("Invalid selection identity or trace hash");}
        let _lock=WRITES.lock().map_err(|_|anyhow::anyhow!("Rule storage lock unavailable"))?;
        let key=Sha256::digest(serde_json::to_vec(&(project,trace))?).iter().map(|b|format!("{b:02x}")).collect::<String>();
        let directory=self.directory.join("selections");std::fs::create_dir_all(&directory)?;let directory=directory.canonicalize()?;
        if !directory.starts_with(&self.directory){bail!("Selection directory escapes rule storage");}
        let path=directory.join(format!("{key}.json"));
        if path.try_exists()?{
            if !std::fs::symlink_metadata(&path)?.file_type().is_file(){bail!("Invalid selection file");}
            let mut bytes=Vec::new();std::fs::File::open(path)?.take(1024*1024+1).read_to_end(&mut bytes)?;
            if bytes.len()>1024*1024{bail!("Selection exceeds read limit");}
            let packet=crate::strict_json::parse(std::str::from_utf8(&bytes)?)?;
            self.validate_selection(&packet,project,trace,trace_hash)?;return Ok(packet);
        }
        let mut files=0;
        for entry in std::fs::read_dir(&directory)?{let entry=entry?;if entry.path().extension().and_then(|s|s.to_str())!=Some("json"){continue;}files+=1;if files>=1000||!entry.file_type()?.is_file(){bail!("Selection storage capacity reached");}}
        let mut identities=std::collections::BTreeSet::new();
        for offset in (0..1000).step_by(100){let catalog=self.list(project,offset,100)?;for row in catalog["rules"].as_array().unwrap(){identities.insert(row["rule"]["id"].as_str().unwrap().to_owned());}if catalog["has_more"]==false{break;}}
        let mut decisions=Vec::new();
        for identity in identities{if let Some(binding)=self.binding(project,&identity)?{if !binding.active{continue;}
            let snapshot=self.get(&binding.rule_sha256)?;let selected=snapshot.rule.selects(project,trace)?;
            decisions.push(serde_json::json!({"binding":binding,"selected":selected}));}}
        let digest=Sha256::digest(serde_json::to_vec(&(project,trace,trace_hash,&decisions))?).iter().map(|b|format!("{b:02x}")).collect::<String>();
        let packet=serde_json::json!({"kind":"online_evaluation_selection","schema_version":2,"project_id":project,"trace_id":trace,"trace_sha256":trace_hash,"decisions":decisions,"selection_sha256":digest,"provider_calls":0,"automatic_execution":false});
        if serde_json::to_vec(&packet)?.len()>1024*1024{bail!("Selection exceeds write limit");}
        self.validate_selection(&packet,project,trace,trace_hash)?;crate::evaluation::commit_new(&path,&packet)?;Ok(packet)
    }
    fn validate_selection(&self,packet:&serde_json::Value,project:&str,trace:&str,trace_hash:&str)->Result<()>{
        if packet.as_object().map(|o|o.len())!=Some(9)||packet["kind"]!="online_evaluation_selection"||packet["schema_version"]!=2||packet["trace_sha256"]!=trace_hash||packet["project_id"]!=project||packet["trace_id"]!=trace||packet["provider_calls"]!=0||packet["automatic_execution"]!=false{bail!("Invalid selection receipt");}
        let decisions=packet["decisions"].as_array().ok_or_else(||anyhow::anyhow!("Missing selection decisions"))?;if decisions.len()>1000{bail!("Too many selection decisions");}
        let digest=Sha256::digest(serde_json::to_vec(&(project,trace,trace_hash,decisions))?).iter().map(|b|format!("{b:02x}")).collect::<String>();if packet["selection_sha256"]!=digest{bail!("Selection integrity mismatch");}
        let mut previous=None;
        for decision in decisions{if decision.as_object().map(|o|o.len())!=Some(2){bail!("Invalid selection decision");}
            let binding:Binding=serde_json::from_value(decision["binding"].clone())?;
            if binding.kind!="online_evaluation_binding"||binding.project_id!=project||!binding.active||binding.version==0||binding.version>1000||binding.binding_sha256!=binding.hash()?||previous.as_ref().is_some_and(|id:&String|id>=&binding.rule_id){bail!("Invalid selected binding");}
            let snapshot=self.get(&binding.rule_sha256)?;
            if snapshot.rule.id!=binding.rule_id||snapshot.rule.project_id!=project||decision["selected"]!=snapshot.rule.selects(project,trace)?{bail!("Selected rule mismatch");}previous=Some(binding.rule_id);
        }Ok(())
    }
    fn list(&self,project:&str,offset:usize,limit:usize)->Result<serde_json::Value>{
        if !technical(project)||offset>1000||!(1..=100).contains(&limit){bail!("Invalid online rule page");}
        let mut rows=Vec::new();let mut count=0;
        for entry in std::fs::read_dir(&self.directory)?{let entry=entry?;let path=entry.path();
            if path.extension().and_then(|s|s.to_str())!=Some("json"){continue;}
            count+=1;if count>1000||!entry.file_type()?.is_file(){bail!("Online rule catalog exceeds scan limit");}
            let hash=path.file_stem().and_then(|s|s.to_str()).ok_or_else(||anyhow::anyhow!("Invalid rule filename"))?;
            let snapshot=self.get(hash)?;if snapshot.rule.project_id!=project{continue;}
            rows.push(serde_json::json!({"rule_sha256":snapshot.rule_sha256,"rule":snapshot.rule}));
        }
        rows.sort_by(|a,b|a["rule_sha256"].as_str().cmp(&b["rule_sha256"].as_str()));
        let total=rows.len();let rules=rows.into_iter().skip(offset).take(limit).collect::<Vec<_>>();
        Ok(serde_json::json!({"kind":"online_evaluation_rule_catalog","project_id":project,"offset":offset,"limit":limit,"total":total,"has_more":offset+rules.len()<total,"order":"rule_sha256_ascending","rules":rules,"provider_calls":0,"automatic_execution":false}))
    }
    fn put(&self,rule:Rule)->Result<Snapshot>{
        let _lock=WRITES.lock().map_err(|_|anyhow::anyhow!("Rule storage lock unavailable"))?;
        let snapshot=Snapshot::new(rule)?;let path=self.path(&snapshot.rule_sha256)?;
        if path.try_exists()? {return self.get(&snapshot.rule_sha256);}
        let mut count=0;
        for entry in std::fs::read_dir(&self.directory)?{let entry=entry?;if entry.path().extension().and_then(|s|s.to_str())!=Some("json"){continue;}
            if !entry.file_type()?.is_file(){bail!("Invalid rule storage entry");}count+=1;if count>=1000{bail!("Online rule storage capacity reached");}}
        crate::evaluation::commit_new(&path,&snapshot)?;Ok(snapshot)
    }
}
pub(crate) async fn save_api(axum::extract::State(app):axum::extract::State<crate::App>,body:axum::body::Bytes)->crate::ApiResult<serde_json::Value>{
    let result=(||->Result<_>{if body.len()>8192{bail!("Rule request exceeds limit");}
        let value=crate::strict_json::parse(std::str::from_utf8(&body)?)?;let rule:Rule=serde_json::from_value(value)?;
        crate::memory::project_exists(&app,&rule.project_id)?;
        crate::model_evaluators::validate_rule(&app.data,&rule)?;
        Ok(serde_json::to_value(Store::open(&app.data)?.put(rule)?)?)
    })();result.map(axum::Json).map_err(|e|crate::error(axum::http::StatusCode::BAD_REQUEST,e))
}
pub(crate) async fn get_api(axum::extract::State(app):axum::extract::State<crate::App>,axum::extract::Path(hash):axum::extract::Path<String>)->crate::ApiResult<serde_json::Value>{
    let result=(||->Result<_>{Ok(serde_json::to_value(Store::open(&app.data)?.get(&hash)?)?)})();
    result.map(axum::Json).map_err(|_|crate::error(axum::http::StatusCode::NOT_FOUND,"Online evaluation rule unavailable"))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Page {project_id:String,#[serde(default)]offset:usize,#[serde(default="page_limit")]limit:usize}
fn page_limit()->usize{20}
pub(crate) async fn list_api(axum::extract::State(app):axum::extract::State<crate::App>,axum::extract::Query(page):axum::extract::Query<Page>)->crate::ApiResult<serde_json::Value>{
    let result=(||->Result<_>{crate::memory::project_exists(&app,&page.project_id)?;Store::open(&app.data)?.list(&page.project_id,page.offset,page.limit)})();
    result.map(axum::Json).map_err(|e|crate::error(axum::http::StatusCode::BAD_REQUEST,e))
}
pub(crate) async fn bind_api(axum::extract::State(app):axum::extract::State<crate::App>,axum::Json(input):axum::Json<BindInput>)->crate::ApiResult<serde_json::Value>{
    let result=(||->Result<_>{let store=Store::open(&app.data)?;let snapshot=store.get(&input.rule_sha256)?;crate::memory::project_exists(&app,&snapshot.rule.project_id)?;Ok(serde_json::to_value(store.bind(input)?)?)})();
    result.map(axum::Json).map_err(|e|crate::error(axum::http::StatusCode::BAD_REQUEST,e))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BindingQuery {project_id:String,rule_id:String}
pub(crate) async fn binding_api(axum::extract::State(app):axum::extract::State<crate::App>,axum::extract::Query(query):axum::extract::Query<BindingQuery>)->crate::ApiResult<serde_json::Value>{
    let result=(||->Result<_>{crate::memory::project_exists(&app,&query.project_id)?;Ok(serde_json::json!({"kind":"online_evaluation_binding_status","project_id":query.project_id,"rule_id":query.rule_id,"binding":Store::open(&app.data)?.binding(&query.project_id,&query.rule_id)?,"provider_calls":0,"automatic_execution":false}))})();
    result.map(axum::Json).map_err(|e|crate::error(axum::http::StatusCode::BAD_REQUEST,e))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SelectionInput {project_id:String,trace_id:String}
pub(crate) async fn select_api(axum::extract::State(app):axum::extract::State<crate::App>,axum::Json(input):axum::Json<SelectionInput>)->crate::ApiResult<serde_json::Value>{
    let result=(||->Result<_>{let fingerprint=app.observability.completed_trace_fingerprint(&input.trace_id,&input.project_id)?;Store::open(&app.data)?.select_trace(&input.project_id,&input.trace_id,&fingerprint)})();
    result.map(axum::Json).map_err(|e|crate::error(axum::http::StatusCode::BAD_REQUEST,e))
}
pub(crate) async fn selection_archive_api(axum::extract::State(app):axum::extract::State<crate::App>,axum::extract::Query(input):axum::extract::Query<SelectionInput>)->crate::ApiResult<serde_json::Value>{
    let result=(||->Result<_>{crate::memory::project_exists(&app,&input.project_id)?;Store::open(&app.data)?.read_selection_archive(&input.project_id,&input.trace_id)})();
    result.map(axum::Json).map_err(|e|crate::error(axum::http::StatusCode::BAD_REQUEST,e))
}
pub(crate) async fn assess_api(axum::extract::State(app):axum::extract::State<crate::App>,axum::Json(input):axum::Json<SelectionInput>)->crate::ApiResult<serde_json::Value>{
    let result=(||->Result<_>{let(hash,health)=app.observability.completed_trace_health(&input.trace_id,&input.project_id)?;Store::open(&app.data)?.assess_trace(&input.project_id,&input.trace_id,&hash,health)})();
    result.map(axum::Json).map_err(|e|crate::error(axum::http::StatusCode::BAD_REQUEST,e))
}
pub(crate) async fn jobs_api(axum::extract::State(app):axum::extract::State<crate::App>,axum::extract::Query(page):axum::extract::Query<Page>)->crate::ApiResult<serde_json::Value>{
    let result=(||->Result<_>{crate::memory::project_exists(&app,&page.project_id)?;jobs_catalog(&app.data,&page.project_id,page.offset,page.limit)})();
    result.map(axum::Json).map_err(|e|crate::error(axum::http::StatusCode::BAD_REQUEST,e))
}
fn read_catalog_packet(path:&std::path::Path)->Result<serde_json::Value>{
    use std::io::Read;
    if !std::fs::symlink_metadata(path)?.file_type().is_file(){bail!("Invalid catalog packet");}
    let mut bytes=Vec::new();std::fs::File::open(path)?.take(8193).read_to_end(&mut bytes)?;if bytes.len()>8192{bail!("Catalog packet exceeds limit");}
    Ok(crate::strict_json::parse(std::str::from_utf8(&bytes)?)?)
}
fn jobs_catalog(data:&std::path::Path,project:&str,offset:usize,limit:usize)->Result<serde_json::Value>{
    if !technical(project)||offset>1000||!(1..=100).contains(&limit){bail!("Invalid job catalog bounds");}
    let _lock=DRAIN.lock().map_err(|_|anyhow::anyhow!("Online worker lock unavailable"))?;
    let directory=data.join("observability/online-rules/jobs");let mut rows=Vec::new();
    if directory.exists(){
        let store=Store::open(data)?;if !directory.canonicalize()?.starts_with(&store.directory){bail!("Job directory escapes storage");}
        let mut count=0;
        for entry in std::fs::read_dir(&directory)?{let path=entry?.path();if path.extension().and_then(|s|s.to_str())!=Some("json"){continue;}count+=1;if count>1000{bail!("Online job scan exceeds limit");}
            let pending=store.process_job(&path,data,false,true)?.is_some();
            let job=read_catalog_packet(&path)?;
            if job["project_id"]!=project{continue;}
            let result=if pending{serde_json::Value::Null}else{read_catalog_packet(&store.directory.join("job-results").join(path.file_name().unwrap()))?};
            rows.push(serde_json::json!({"status":if pending{"pending"}else{result["status"].as_str().unwrap()},"job":job,"result":result}));
        }
    }
    rows.sort_by(|a,b|a["job"]["selection_sha256"].as_str().cmp(&b["job"]["selection_sha256"].as_str()));let total=rows.len();let jobs=rows.into_iter().skip(offset).take(limit).collect::<Vec<_>>();
    Ok(serde_json::json!({"kind":"online_evaluation_job_catalog","project_id":project,"offset":offset,"limit":limit,"total":total,"has_more":offset+jobs.len()<total,"order":"selection_sha256_ascending","jobs":jobs,"provider_calls":0,"automatic_execution":false}))
}
static DRAIN:std::sync::Mutex<()>=std::sync::Mutex::new(());
pub(crate) fn drain_jobs(data:&std::path::Path,limit:usize)->Result<serde_json::Value>{
    drain_jobs_origin(data,limit,false)
}
fn drain_jobs_origin(data:&std::path::Path,limit:usize,automatic:bool)->Result<serde_json::Value>{
    if !(1..=100).contains(&limit){bail!("Invalid online job batch limit");}
    let _lock=DRAIN.lock().map_err(|_|anyhow::anyhow!("Online worker lock unavailable"))?;let store=Store::open(data)?;let directory=store.directory.join("jobs");let mut paths=Vec::new();
    if directory.exists(){if !directory.canonicalize()?.starts_with(&store.directory){bail!("Job directory escapes storage");}for entry in std::fs::read_dir(&directory)?{let entry=entry?;if entry.path().extension().and_then(|s|s.to_str())!=Some("json"){continue;}paths.push(entry.path());if paths.len()>1000{bail!("Online job scan exceeds limit");}}}
    paths.sort();let mut completed=0;let mut failed=0;let mut already_finished=0;
    for path in paths{if completed+failed==limit{break;}match store.process_job(&path,data,automatic,false)?{Some(true)=>completed+=1,Some(false)=>failed+=1,None=>already_finished+=1}}
    Ok(serde_json::json!({"kind":"online_evaluation_job_batch","completed":completed,"failed":failed,"already_finished":already_finished,"provider_calls":0,"automatic_execution":automatic}))
}
/// Owned by Studio: dropping the guard stops polling. An admitted bounded batch may finish.
pub(crate) struct Worker(tokio::task::JoinHandle<()>);
impl Drop for Worker { fn drop(&mut self){self.0.abort();} }
pub(crate) fn spawn_worker(data:std::sync::Arc<std::path::PathBuf>)->Worker{
    Worker(tokio::spawn(async move {
        let mut last_error=None;
        loop {
            if data.join("observability/online-rules/jobs").exists(){
                let root=data.clone();
                let result=tokio::task::spawn_blocking(move ||drain_jobs_origin(&root,20,true)).await;
                let error=match result{Ok(Ok(_))=>None,Ok(Err(e))=>Some(e.to_string()),Err(e)=>Some(e.to_string())};
                if error!=last_error{if let Some(message)=&error{eprintln!("Online evaluation worker failed: {message}");}last_error=error;}
            }
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DrainInput {limit:usize}
pub(crate) async fn drain_api(axum::extract::State(app):axum::extract::State<crate::App>,axum::Json(input):axum::Json<DrainInput>)->crate::ApiResult<serde_json::Value>{
    drain_jobs(&app.data,input.limit).map(axum::Json).map_err(|e|crate::error(axum::http::StatusCode::BAD_REQUEST,e))
}
pub(crate) fn selection_pin(data:&std::path::Path,project:&str,trace:&str,trace_hash:&str)->Result<String>{
    let selection=Store::open(data)?.select_trace(project,trace,trace_hash)?;Ok(selection["selection_sha256"].as_str().unwrap().into())
}
pub(crate) fn model_selections(data:&std::path::Path,project:&str,trace:&str,trace_hash:&str)->Result<Vec<serde_json::Value>>{
    let store=Store::open(data)?;let selection=store.select_trace(project,trace,trace_hash)?;let mut rows=Vec::new();
    for decision in selection["decisions"].as_array().unwrap(){if decision["selected"]!=true{continue;}let binding:Binding=serde_json::from_value(decision["binding"].clone())?;let snapshot=store.get(&binding.rule_sha256)?;
        if let Some(hash)=snapshot.rule.evaluator_id.strip_prefix("model_quality."){if snapshot.rule.evaluator_version!=1{bail!("Unsupported model evaluator version");}rows.push(serde_json::json!({"selection_sha256":selection["selection_sha256"],"binding_sha256":binding.binding_sha256,"rule_sha256":binding.rule_sha256,"evaluator_sha256":hash}));}
    }Ok(rows)
}
/// Trace completion hook: admission failures never change the primary trace result.
pub(crate) fn enqueue_completed(data:&std::path::Path,project:&str,trace:&str)->Result<()>{
    if !data.join("observability/online-rules").exists(){return Ok(());}
    let hash=crate::observability::Store::new(data)?.completed_trace_fingerprint(trace,project)?;
    Store::open(data)?.enqueue_trace(project,trace,&hash)?;Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn rule()->Rule {Rule{id:"quality".into(),project_id:"default".into(),evaluator_id:"rubric".into(),evaluator_version:1,sample_rate:0.5,enabled:true}}
    #[tokio::test]
    async fn worker_processes_retained_jobs_and_preserves_first_execution_origin(){
        let root=std::env::temp_dir().join(format!("allpaka-worker-{}",crate::evaluation::new_id()));std::fs::create_dir_all(&root).unwrap();let root=root.canonicalize().unwrap();let store=Store::open(&root).unwrap();
        let mut rule=rule();rule.evaluator_id="trace_health".into();rule.sample_rate=1.0;let snapshot=store.put(rule).unwrap();store.bind(BindInput{rule_sha256:snapshot.rule_sha256,base_version:0,active:true}).unwrap();
        let traces=crate::observability::Store::new(&root).unwrap();let trace=traces.begin("worker","default").unwrap();trace.span("turn","agent",None).finish("completed",&serde_json::Value::Null);
        let job=std::fs::read_dir(store.directory.join("jobs")).unwrap().next().unwrap().unwrap().path();let result_path=store.directory.join("job-results").join(job.file_name().unwrap());
        assert!(!result_path.exists());let pending=jobs_catalog(&root,"default",0,1).unwrap();assert_eq!(pending["total"],1);assert_eq!(pending["jobs"][0]["status"],"pending");assert!(!store.directory.join("job-results").exists());
        assert_eq!(jobs_catalog(&root,"other",0,1).unwrap()["total"],0);assert!(jobs_catalog(&root,"default",0,0).is_err());
        let worker=spawn_worker(std::sync::Arc::new(root.clone()));
        tokio::time::timeout(std::time::Duration::from_secs(10),async {while !result_path.exists(){tokio::time::sleep(std::time::Duration::from_millis(20)).await;}}).await.unwrap();
        drop(worker);
        let bytes=std::fs::read(&result_path).unwrap();let result:serde_json::Value=serde_json::from_slice(&bytes).unwrap();assert_eq!(result["status"],"completed");assert_eq!(result["automatic_execution"],true);assert_eq!(result["provider_calls"],0);
        let catalog=jobs_catalog(&root,"default",0,1).unwrap();assert_eq!(catalog["jobs"][0]["result"],result);assert_eq!(catalog["jobs"][0]["status"],"completed");assert_eq!(jobs_catalog(&root,"default",1,1).unwrap()["jobs"],serde_json::json!([]));
        let mut corrupt=result.clone();corrupt["status"]=serde_json::json!("failed");std::fs::write(&result_path,serde_json::to_vec(&corrupt).unwrap()).unwrap();assert!(jobs_catalog(&root,"default",0,1).is_err());std::fs::write(&result_path,&bytes).unwrap();
        assert_eq!(drain_jobs(&root,20).unwrap()["already_finished"],1);assert_eq!(std::fs::read(&result_path).unwrap(),bytes);
        let(hash,health)=traces.completed_trace_health(&trace.id(),"default").unwrap();assert_eq!(store.assess_trace("default",&trace.id(),&hash,health).unwrap()["automatic_execution"],true);
        let restarted=spawn_worker(std::sync::Arc::new(root.clone()));tokio::time::sleep(std::time::Duration::from_millis(100)).await;drop(restarted);assert_eq!(std::fs::read(&result_path).unwrap(),bytes);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn changed_job_evidence_records_failure_without_assessment_or_replay(){
        let root=std::env::temp_dir().join(format!("allpaka-job-failure-{}",crate::evaluation::new_id()));std::fs::create_dir_all(&root).unwrap();let root=root.canonicalize().unwrap();let store=Store::open(&root).unwrap();
        let mut rule=rule();rule.evaluator_id="trace_health".into();rule.sample_rate=1.0;let snapshot=store.put(rule).unwrap();store.bind(BindInput{rule_sha256:snapshot.rule_sha256,base_version:0,active:true}).unwrap();
        let traces=crate::observability::Store::new(&root).unwrap();let trace=traces.begin("changed","default").unwrap();trace.span("turn","agent",None).finish("completed",&serde_json::Value::Null);
        trace.span("tool","changed",Some(0)).finish("completed",&serde_json::Value::Null);
        let result=drain_jobs(&root,20).unwrap();assert_eq!(result["failed"],1);assert_eq!(result["completed"],0);assert!(!store.directory.join("assessments").exists());
        assert_eq!(drain_jobs(&root,20).unwrap()["already_finished"],1);std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn completed_native_trace_automatically_admits_one_durable_job(){
        let root=std::env::temp_dir().join(format!("allpaka-auto-job-{}",crate::evaluation::new_id()));std::fs::create_dir_all(&root).unwrap();let root=root.canonicalize().unwrap();let store=Store::open(&root).unwrap();
        let mut rule=rule();rule.evaluator_id="trace_health".into();rule.sample_rate=1.0;let snapshot=store.put(rule).unwrap();store.bind(BindInput{rule_sha256:snapshot.rule_sha256,base_version:0,active:true}).unwrap();
        let traces=crate::observability::Store::new(&root).unwrap();let trace=traces.begin("automatic","default").unwrap();let mut guard=trace.span("turn","agent",None);
        assert!(!store.directory.join("jobs").exists());guard.finish("completed",&serde_json::Value::Null);
        let jobs=std::fs::read_dir(store.directory.join("jobs")).unwrap().collect::<Vec<_>>();assert_eq!(jobs.len(),1);
        enqueue_completed(&root,"default",&trace.id()).unwrap();assert_eq!(std::fs::read_dir(store.directory.join("jobs")).unwrap().count(),1);
        let job:serde_json::Value=serde_json::from_slice(&std::fs::read(jobs[0].as_ref().unwrap().path()).unwrap()).unwrap();assert_eq!(job["status"],"pending");assert_eq!(job["trace_id"],trace.id());assert_eq!(job["provider_calls"],0);
        let batch=drain_jobs(&root,20).unwrap();assert_eq!(batch["completed"],1);assert_eq!(batch["failed"],0);
        let repeated=drain_jobs(&root,20).unwrap();assert_eq!(repeated["completed"],0);assert_eq!(repeated["already_finished"],1);
        let assessment_path=store.directory.join("assessments").join(format!("{}.json",job["selection_sha256"].as_str().unwrap()));let original=std::fs::read(&assessment_path).unwrap();let mut changed:serde_json::Value=serde_json::from_slice(&original).unwrap();changed["assessments"][0]["scores"]["span_success_rate"]=serde_json::json!(0.0);std::fs::write(&assessment_path,serde_json::to_vec(&changed).unwrap()).unwrap();assert!(drain_jobs(&root,20).is_err());std::fs::write(&assessment_path,original).unwrap();assert_eq!(drain_jobs(&root,20).unwrap()["already_finished"],1);

        let failed=traces.begin("failed","default").unwrap();failed.span("turn","agent",None).finish("failed",&serde_json::Value::Null);assert_eq!(std::fs::read_dir(store.directory.join("jobs")).unwrap().count(),1);
        for index in 0..999{std::fs::write(store.directory.join("jobs").join(format!("quota-{index}.json")),b"{}").unwrap();}
        let at_capacity=traces.begin("queue-full","default").unwrap();at_capacity.span("turn","agent",None).finish("completed",&serde_json::Value::Null);
        assert!(traces.completed_trace_fingerprint(&at_capacity.id(),"default").is_ok());assert_eq!(std::fs::read_dir(store.directory.join("jobs")).unwrap().count(),1000);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn native_assessments_are_pinned_idempotent_and_unknown_evaluators_fail(){
        let root=std::env::temp_dir().join(format!("allpaka-assessment-{}",crate::evaluation::new_id()));std::fs::create_dir_all(&root).unwrap();let root=root.canonicalize().unwrap();let store=Store::open(&root).unwrap();
        let mut health=rule();health.evaluator_id="trace_health".into();health.sample_rate=1.0;let snapshot=store.put(health).unwrap();store.bind(BindInput{rule_sha256:snapshot.rule_sha256,base_version:0,active:true}).unwrap();
        let scored=store.assess_trace("default","trace-1",&"a".repeat(64),0.5).unwrap();assert_eq!(scored["assessments"][0]["scores"]["span_success_rate"],0.5);assert_eq!(scored["assessments"][0]["status"],"completed");
        assert_eq!(Store::open(&root).unwrap().assess_trace("default","trace-1",&"a".repeat(64),0.5).unwrap(),scored);assert!(store.assess_trace("default","trace-1",&"b".repeat(64),0.5).is_err());
        let mut unsupported=rule();unsupported.sample_rate=1.0;let snapshot=store.put(unsupported).unwrap();store.bind(BindInput{rule_sha256:snapshot.rule_sha256,base_version:1,active:true}).unwrap();
        let failed=store.assess_trace("default","trace-2",&"b".repeat(64),1.0).unwrap();assert_eq!(failed["assessments"][0]["status"],"failed");assert_eq!(failed["assessments"][0]["error_code"],"evaluator_unavailable");assert_eq!(failed["assessments"][0]["scores"],serde_json::json!({}));std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn selections_retain_binding_versions_and_do_not_reselect_old_traces(){
        let root=std::env::temp_dir().join(format!("allpaka-selection-{}",crate::evaluation::new_id()));std::fs::create_dir_all(&root).unwrap();let root=root.canonicalize().unwrap();let store=Store::open(&root).unwrap();
        let mut full=rule();full.sample_rate=1.0;let snapshot=store.put(full).unwrap();
        store.bind(BindInput{rule_sha256:snapshot.rule_sha256.clone(),base_version:0,active:true}).unwrap();
        let selected=store.select_trace("default","trace-1",&"a".repeat(64)).unwrap();assert_eq!(selected["decisions"][0]["selected"],true);assert_eq!(selected["decisions"][0]["binding"]["version"],1);
        store.bind(BindInput{rule_sha256:snapshot.rule_sha256,base_version:1,active:false}).unwrap();
        assert_eq!(store.select_trace("default","trace-1",&"a".repeat(64)).unwrap(),selected);
        assert!(store.select_trace("default","trace-1",&"b".repeat(64)).is_err());
        assert_eq!(store.select_trace("default","trace-2",&"b".repeat(64)).unwrap()["decisions"],serde_json::json!([]));
        assert_eq!(Store::open(&root).unwrap().select_trace("default","trace-1",&"a".repeat(64)).unwrap(),selected);
        let mut altered=selected.clone();altered["decisions"][0]["selected"]=serde_json::json!(false);assert!(store.validate_selection(&altered,"default","trace-1",&"a".repeat(64)).is_err());
        assert!(store.validate_selection(&selected,"other","trace-1",&"a".repeat(64)).is_err());
        let archived=store.read_selection_archive("default","trace-1").unwrap();assert_eq!(archived["trace_evidence_pinned"],true);assert_eq!(archived["selection"],selected);
        let mut legacy=selected.clone();legacy.as_object_mut().unwrap().remove("trace_sha256");legacy["schema_version"]=serde_json::json!(1);
        legacy["selection_sha256"]=serde_json::json!(Sha256::digest(serde_json::to_vec(&("default","trace-1",legacy["decisions"].as_array().unwrap())).unwrap()).iter().map(|b|format!("{b:02x}")).collect::<String>());
        let key=Sha256::digest(serde_json::to_vec(&("default","trace-1")).unwrap()).iter().map(|b|format!("{b:02x}")).collect::<String>();let path=store.directory.join("selections").join(format!("{key}.json"));let bytes=serde_json::to_vec(&legacy).unwrap();std::fs::write(&path,&bytes).unwrap();
        let archived=store.read_selection_archive("default","trace-1").unwrap();assert_eq!(archived["trace_evidence_pinned"],false);assert_eq!(archived["execution_eligible"],false);assert_eq!(archived["selection"],legacy);assert_eq!(std::fs::read(&path).unwrap(),bytes);
        assert!(store.select_trace("default","trace-1",&"a".repeat(64)).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn bindings_pin_snapshots_and_reject_stale_changes_or_tampering(){
        let root=std::env::temp_dir().join(format!("allpaka-bindings-{}",crate::evaluation::new_id()));std::fs::create_dir_all(&root).unwrap();let root=root.canonicalize().unwrap();
        let store=Store::open(&root).unwrap();let first=store.put(rule()).unwrap();
        assert!(store.binding("default","missing").unwrap().is_none());assert!(!store.directory.join("bindings").exists());
        let active=store.bind(BindInput{rule_sha256:first.rule_sha256.clone(),base_version:0,active:true}).unwrap();assert_eq!(active.version,1);
        assert!(store.bind(BindInput{rule_sha256:first.rule_sha256.clone(),base_version:0,active:false}).is_err());
        let mut next=rule();next.evaluator_version=2;let next=store.put(next).unwrap();
        let switched=store.bind(BindInput{rule_sha256:next.rule_sha256.clone(),base_version:1,active:true}).unwrap();assert_eq!(switched.version,2);assert_eq!(switched.rule_sha256,next.rule_sha256);
        let inactive=store.bind(BindInput{rule_sha256:next.rule_sha256.clone(),base_version:2,active:false}).unwrap();assert_eq!(inactive.version,3);assert!(!inactive.active);
        assert_eq!(Store::open(&root).unwrap().binding("default","quality").unwrap().unwrap().binding_sha256,inactive.binding_sha256);
        let mut disabled=rule();disabled.enabled=false;let disabled=store.put(disabled).unwrap();assert!(store.bind(BindInput{rule_sha256:disabled.rule_sha256,base_version:3,active:true}).is_err());
        let directory=store.binding_directory("default","quality").unwrap();let mut tampered=serde_json::to_value(&inactive).unwrap();tampered["active"]=serde_json::json!(true);std::fs::write(directory.join("3.json"),serde_json::to_vec(&tampered).unwrap()).unwrap();assert!(store.binding("default","quality").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn immutable_rule_storage_survives_reopen_and_rejects_tampering(){
        let root=std::env::temp_dir().join(format!("allpaka-online-rules-{}",crate::evaluation::new_id()));
        std::fs::create_dir_all(&root).unwrap();let root=root.canonicalize().unwrap();
        let store=Store::open(&root).unwrap();let snapshot=store.put(rule()).unwrap();
        assert_eq!(store.put(rule()).unwrap().rule_sha256,snapshot.rule_sha256);
        let restored=Store::open(&root).unwrap().get(&snapshot.rule_sha256).unwrap();assert_eq!(restored.rule.evaluator_version,1);
        let mut revision=rule();revision.evaluator_version=2;let next=store.put(revision).unwrap();assert_ne!(next.rule_sha256,snapshot.rule_sha256);
        assert_eq!(store.get(&snapshot.rule_sha256).unwrap().rule.evaluator_version,1);
        let page=store.list("default",0,1).unwrap();assert_eq!(page["total"],2);assert_eq!(page["has_more"],true);
        let second=store.list("default",1,1).unwrap();assert_eq!(second["has_more"],false);assert_ne!(page["rules"][0]["rule_sha256"],second["rules"][0]["rule_sha256"]);
        assert_eq!(store.list("other",0,20).unwrap()["total"],0);assert!(store.list("default",0,101).is_err());
        let path=store.path(&snapshot.rule_sha256).unwrap();let mut tampered=serde_json::to_value(&snapshot).unwrap();tampered["rule"]["sample_rate"]=serde_json::json!(1.0);
        std::fs::write(&path,serde_json::to_vec(&tampered).unwrap()).unwrap();assert!(store.get(&snapshot.rule_sha256).is_err());assert!(store.put(rule()).is_err());assert!(store.list("default",0,20).is_err());
        assert!(store.get("../escape").is_err());std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn selection_is_pinned_reproducible_and_scoped(){
        let original=rule();let restored:Rule=serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
        let selected=(0..1000).map(|i|original.selects("default",&format!("trace-{i}")).unwrap()).collect::<Vec<_>>();
        assert!(selected.iter().filter(|v|**v).count()>400&&selected.iter().filter(|v|**v).count()<600);
        for i in (0..1000).rev(){assert_eq!(selected[i],restored.selects("default",&format!("trace-{i}")).unwrap());}
        assert!(!original.selects("other","trace-1").unwrap());
        let mut revision=original.clone();revision.evaluator_version=2;
        assert!((0..1000).any(|i|selected[i]!=revision.selects("default",&format!("trace-{i}")).unwrap()));
        revision.enabled=false;assert!(!revision.selects("default","trace-1").unwrap());
        revision.enabled=true;revision.sample_rate=0.0;assert!(!revision.selects("default","trace-1").unwrap());
        revision.sample_rate=1.0;assert!(revision.selects("default","trace-1").unwrap());
    }
    #[test]
    fn malformed_rules_fail_before_selection(){
        for rate in [f64::NAN,f64::INFINITY,-0.1,1.1]{let mut candidate=rule();candidate.sample_rate=rate;assert!(candidate.selects("default","trace").is_err());}
        let mut candidate=rule();candidate.evaluator_version=0;assert!(candidate.validate().is_err());
        candidate=rule();candidate.id="../escape".into();assert!(candidate.validate().is_err());
        assert!(rule().selects("default","../trace").is_err());
        assert!(serde_json::from_value::<Rule>(serde_json::json!({"id":"q","project_id":"p","evaluator_id":"e","evaluator_version":1,"sample_rate":0.5,"enabled":true,"prompt":"untrusted"})).is_err());
    }
}
