use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Rule {
    pub id: String,
    pub kind: String,
    pub value: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub kind: String,
    pub schema_version: u32,
    pub policy_sha256: String,
    pub rules: Vec<Rule>,
}

impl Manifest {
    pub fn create(rules: Vec<Rule>) -> Result<Self> {
        if !(1..=32).contains(&rules.len()) { bail!("Use 1-32 guardrail rules"); }
        let mut ids = HashSet::new();
        let mut canonical = Vec::new();
        for rule in &rules {
            if rule.id.is_empty() || rule.id.len()>80 || !rule.id.bytes().all(|b|b.is_ascii_alphanumeric() || b"._-".contains(&b)) || !ids.insert(&rule.id) { bail!("Invalid or duplicate technical rule ID"); }
            match rule.kind.as_str() {
                "min_bytes" | "max_bytes" => {
                    if !rule.value.as_u64().is_some_and(|n|n<=64000) { bail!("Invalid byte limit"); }
                }
                "json_valid" => { if rule.value != Value::Bool(true) { bail!("JSON validation requires true"); } }
                "forbidden_substrings" | "required_substrings" => {
                    let values=rule.value.as_array().context("Fragments must be an array")?;
                    if !(1..=100).contains(&values.len()) || values.iter().any(|v|!v.as_str().is_some_and(|s|!s.is_empty() && s.len()<=1000)) { bail!("Invalid literal fragments"); }
                }
                _ => bail!("Unsupported guardrail kind"),
            }
            canonical.push(json!([rule.id,rule.kind,rule.value]));
        }
        let bytes=serde_json::to_vec(&canonical)?;
        if bytes.len()>128*1024 { bail!("Policy exceeds 128 KiB"); }
        let policy_sha256=Sha256::digest(&bytes).iter().map(|b|format!("{b:02x}")).collect();
        Ok(Self{kind:"local_guardrail_policy".into(),schema_version:1,policy_sha256,rules})
    }

    pub fn check(&self,text:&str,stage:&str,action:&str)->Result<Value> {
        self.validate()?;
        if !["input","output"].contains(&stage) || !["observe","block"].contains(&action) {bail!("Invalid guardrail stage or action");}
        if text.len()>64000 {bail!("Guardrail text exceeds 64000 UTF-8 bytes");}
        let rules=self.rules.iter().map(|rule| {
            let passed=match rule.kind.as_str() {
                "min_bytes"=>text.len() as u64>=rule.value.as_u64().unwrap(),
                "max_bytes"=>text.len() as u64<=rule.value.as_u64().unwrap(),
                "required_substrings"=>rule.value.as_array().unwrap().iter().all(|value|text.contains(value.as_str().unwrap())),
                "forbidden_substrings"=>!rule.value.as_array().unwrap().iter().any(|value|text.contains(value.as_str().unwrap())),
                "json_valid"=>crate::strict_json::parse(text).is_ok(),
                _=>unreachable!("Validated rule kind"),
            };
            json!({"rule_id":rule.id,"kind":rule.kind,"passed":passed})
        }).collect::<Vec<_>>();
        let passed=rules.iter().all(|rule|rule["passed"]==true);
        Ok(json!({"kind":"local_guardrail","schema_version":1,"stage":stage,"action":action,"passed":passed,"blocked":!passed&&action=="block","rules":rules,"policy_sha256":self.policy_sha256,"provider_calls":0,"content_captured":false}))
    }

    pub fn validate(&self) -> Result<()> {
        if self.kind!="local_guardrail_policy" || self.schema_version!=1 { bail!("Invalid policy schema"); }
        let expected=Self::create(self.rules.clone())?;
        if expected.policy_sha256!=self.policy_sha256 { bail!("Policy fingerprint mismatch"); }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rules_and_tampering() {
        let mut manifest=Manifest::create(vec![Rule{id:"json".into(),kind:"json_valid".into(),value:json!(true)}]).unwrap();
        manifest.validate().unwrap();
        manifest.rules[0].value=json!(false);assert!(manifest.validate().is_err());
        for value in [json!(true),json!(-1),json!(64001),json!(1.0)] {
            assert!(Manifest::create(vec![Rule{id:"limit".into(),kind:"max_bytes".into(),value}]).is_err());
        }
        let rule=Rule{id:"duplicate".into(),kind:"min_bytes".into(),value:json!(0)};
        assert!(Manifest::create(vec![rule.clone(),rule]).is_err());
    }
    #[test]
    fn persistent_store_roundtrip_and_corruption() {
        let root=std::env::temp_dir().join(format!("allpaka-policy-store-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let store=Store::open(&root).unwrap();
        let manifest:Manifest=serde_json::from_str(include_str!("../testdata/guardrail-policy-sdk.json")).unwrap();
        store.put(&manifest).unwrap();store.put(&manifest).unwrap();
        let catalog=store.list(0,20).unwrap();assert_eq!(catalog["total"],1);assert_eq!(catalog["policies"][0]["rule_count"],2);assert!(catalog["policies"][0].get("rules").is_none());
        assert_eq!(store.list(1,20).unwrap()["policies"],json!([]));assert!(store.list(0,0).is_err());assert!(store.list(10001,20).is_err());
        let reopened=Store::open(&root).unwrap();assert_eq!(reopened.get(&manifest.policy_sha256).unwrap().policy_sha256,manifest.policy_sha256);
        assert!(store.get("../escape").is_err());
        std::fs::write(store.path(&manifest.policy_sha256).unwrap(),b"{\"kind\":1,\"kind\":2}").unwrap();
        assert!(store.get(&manifest.policy_sha256).is_err());assert!(store.put(&manifest).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn required_fragments_match_all_literals_and_keep_receipts_private() {
        let manifest=Manifest::create(vec![Rule{id:"required".into(),kind:"required_substrings".into(),value:json!(["Привет","ready"])}]).unwrap();
        assert_eq!(manifest.check("Привет ready","output","block").unwrap()["passed"],true);
        for text in ["Привет","привет ready","ready"] {
            let receipt=manifest.check(text,"output","block").unwrap();
            assert_eq!(receipt["blocked"],true);
            assert_eq!(receipt["rules"][0]["kind"],"required_substrings");
            assert!(!receipt.to_string().contains("Привет"));
            assert_eq!(crate::observability::public_usage(&json!({"guardrail_receipt":receipt}))["guardrail_receipt"]["rules"][0]["kind"],"required_substrings");
        }
        assert_eq!(manifest.check("ready","input","observe").unwrap()["blocked"],false);
        for value in [json!([]),json!([""]),json!([true])] {
            assert!(Manifest::create(vec![Rule{id:"required".into(),kind:"required_substrings".into(),value}]).is_err());
        }
    }
    #[test]
    fn native_checks_byte_boundaries_strict_json_and_private_receipts() {
        let length=Manifest::create(vec![Rule{id:"limit".into(),kind:"max_bytes".into(),value:json!(4)}]).unwrap();
        assert_eq!(length.check("яя","input","block").unwrap()["passed"],true);
        assert_eq!(length.check("яяX","input","block").unwrap()["blocked"],true);
        assert_eq!(length.check("яяX","output","observe").unwrap()["blocked"],false);
        let manifest=Manifest::create(vec![Rule{id:"json".into(),kind:"json_valid".into(),value:json!(true)}]).unwrap();
        for text in ["NaN","1e999","{\"a\":1,\"a\":2}","{\"nested\":{\"a\":1,\"a\":2}}"] {assert_eq!(manifest.check(text,"output","block").unwrap()["blocked"],true);}
        assert_eq!(manifest.check("{\"a\":[true,null,1]}","output","block").unwrap()["passed"],true);
        assert!(manifest.check(&"x".repeat(64001),"input","observe").is_err());
        assert!(manifest.check("{}","unknown","block").is_err());
        let private=Manifest::create(vec![Rule{id:"literal".into(),kind:"forbidden_substrings".into(),value:json!(["PRIVATE"])}]).unwrap();
        let receipt=private.check("PRIVATE","input","block").unwrap();assert_eq!(receipt["blocked"],true);assert!(!receipt.to_string().contains("PRIVATE"));
    }
    #[test]
    fn prepared_pair_pins_rules_before_execution() {
        let root=std::env::temp_dir().join(format!("allpaka-policy-pair-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let store=Store::open(&root).unwrap();let manifest=Manifest::create(vec![Rule{id:"limit".into(),kind:"max_bytes".into(),value:json!(4)}]).unwrap();store.put(&manifest).unwrap();
        let selection=Selection{input_policy_sha256:manifest.policy_sha256.clone(),output_policy_sha256:manifest.policy_sha256.clone(),action:"block".into()};let prepared=selection.prepare(&store).unwrap();
        std::fs::write(store.path(&manifest.policy_sha256).unwrap(),b"corrupt").unwrap();
        assert_eq!(prepared.input("longer").unwrap()["blocked"],true);assert_eq!(prepared.output("safe").unwrap()["passed"],true);assert!(prepared.buffers_output());assert!(selection.prepare(&store).is_err());
        let mut settings:crate::types::Settings=serde_json::from_value(json!({"provider":"local","model":"mock"})).unwrap();
        let original=serde_json::to_value(&settings).unwrap();settings.prepared_guardrails=Some(std::sync::Arc::new(prepared));
        assert_eq!(serde_json::to_value(&settings).unwrap(),original);
        let restored:crate::types::Settings=serde_json::from_value(original).unwrap();assert!(restored.prepared_guardrails.is_none());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn sdk_fingerprint_and_strict_schema() {
        let manifest:Manifest=serde_json::from_str(include_str!("../testdata/guardrail-policy-sdk.json")).unwrap();
        manifest.validate().unwrap();
        let mut value=serde_json::to_value(&manifest).unwrap();value["schema_version"]=json!(true);
        assert!(serde_json::from_value::<Manifest>(value).is_err());
        let mut value=serde_json::to_value(&manifest).unwrap();value["extra"]=json!(1);
        assert!(serde_json::from_value::<Manifest>(value).is_err());
    }
}

pub(crate) struct Store { root: std::path::PathBuf }
impl Store {
    pub fn open(data:&std::path::Path)->Result<Self> {
        let root=data.join("guardrail-policies");
        std::fs::create_dir_all(&root)?;
        if std::fs::symlink_metadata(&root)?.file_type().is_symlink() { bail!("Policy directory cannot be a symlink"); }
        Ok(Self{root})
    }
    fn path(&self, hash:&str)->Result<std::path::PathBuf> {
        if hash.len()!=64 || !hash.bytes().all(|b|b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) { bail!("Invalid policy fingerprint"); }
        if std::fs::symlink_metadata(&self.root)?.file_type().is_symlink() { bail!("Policy directory cannot be a symlink"); }
        Ok(self.root.join(format!("{hash}.json")))
    }
    pub fn get(&self,hash:&str)->Result<Manifest> {
        use std::io::Read;
        let path=self.path(hash)?;
        if !std::fs::symlink_metadata(&path)?.file_type().is_file() { bail!("Policy must be a regular file"); }
        let mut bytes=Vec::new();std::fs::File::open(path)?.take(128*1024+1).read_to_end(&mut bytes)?;
        if bytes.len()>128*1024 { bail!("Policy file exceeds 128 KiB"); }
        let value=crate::strict_json::parse(std::str::from_utf8(&bytes)?)?;
        let manifest:Manifest=serde_json::from_value(value)?;manifest.validate()?;
        if manifest.policy_sha256!=hash { bail!("Policy filename mismatch"); }
        Ok(manifest)
    }
    pub fn list(&self,offset:usize,limit:usize)->Result<Value> {
        if offset>10000 || !(1..=100).contains(&limit) {bail!("Invalid policy page");}
        self.path(&"0".repeat(64))?;
        let mut hashes=std::collections::BTreeSet::new();
        let mut scanned=0;
        for entry in std::fs::read_dir(&self.root)? {
            scanned+=1;if scanned>10000 {bail!("Policy catalog scan limit exceeded");}
            let entry=entry?;let name=entry.file_name();let Some(name)=name.to_str() else {continue};
            let Some(hash)=name.strip_suffix(".json") else {continue};
            self.path(hash)?;hashes.insert(hash.to_owned());
        }
        let total=hashes.len();let mut policies=Vec::new();
        for hash in hashes.into_iter().skip(offset).take(limit) {
            let manifest=self.get(&hash)?;
            policies.push(json!({"policy_sha256":hash,"rule_count":manifest.rules.len(),"schema_version":manifest.schema_version}));
        }
        Ok(json!({"policies":policies,"total":total,"offset":offset,"limit":limit,"order":"policy_sha256_ascending","provider_calls":0}))
    }
    pub fn put(&self,manifest:&Manifest)->Result<Manifest> {
        use std::io::Write;
        static NEXT:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
        manifest.validate()?;let path=self.path(&manifest.policy_sha256)?;
        let bytes=serde_json::to_vec(manifest)?;if bytes.len()>128*1024 { bail!("Policy file exceeds 128 KiB"); }
        let sequence=NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
        let temp=self.root.join(format!(".{}-{sequence}.tmp",std::process::id()));
        let mut options=std::fs::OpenOptions::new();options.write(true).create_new(true);
        #[cfg(unix)] { use std::os::unix::fs::OpenOptionsExt;options.mode(0o600); }
        let mut file=options.open(&temp)?;
        let written=(||->Result<()> {file.write_all(&bytes)?;file.sync_all()?;
            match std::fs::hard_link(&temp,&path) {Ok(())=>{},Err(error) if error.kind()==std::io::ErrorKind::AlreadyExists=>{},Err(error)=>return Err(error.into())} Ok(())})();
        drop(file);let cleanup=std::fs::remove_file(&temp);written?;cleanup?;
        self.get(&manifest.policy_sha256)
    }
}

pub(crate) async fn save_api(axum::extract::State(app):axum::extract::State<crate::App>,body:axum::body::Bytes)->crate::ApiResult<Value> {
    let result=(||->Result<Value>{
        if body.len()>128*1024 {bail!("Policy file exceeds 128 KiB");}
        let value=crate::strict_json::parse(std::str::from_utf8(&body)?)?;
        let manifest:Manifest=serde_json::from_value(value)?;
        let saved=Store::open(&app.data)?.put(&manifest)?;
        Ok(json!({"policy":saved,"provider_calls":0}))
    })();
    result.map(axum::Json).map_err(|error|crate::error(axum::http::StatusCode::BAD_REQUEST,error))
}
pub(crate) async fn get_api(axum::extract::State(app):axum::extract::State<crate::App>,axum::extract::Path(hash):axum::extract::Path<String>)->crate::ApiResult<Value> {
    let result=(||->Result<Value>{Ok(json!({"policy":Store::open(&app.data)?.get(&hash)?,"provider_calls":0}))})();
    result.map(axum::Json).map_err(|error|crate::error(axum::http::StatusCode::BAD_REQUEST,error))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Page {#[serde(default)] offset:usize,#[serde(default="page_limit")] limit:usize}
fn page_limit()->usize {20}
pub(crate) async fn list_api(axum::extract::State(app):axum::extract::State<crate::App>,axum::extract::Query(page):axum::extract::Query<Page>)->crate::ApiResult<Value> {
    Store::open(&app.data).and_then(|store|store.list(page.offset,page.limit)).map(axum::Json).map_err(|error|crate::error(axum::http::StatusCode::BAD_REQUEST,error))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Create { rules:Vec<Rule> }
pub(crate) async fn create_api(axum::extract::State(app):axum::extract::State<crate::App>,body:axum::body::Bytes)->crate::ApiResult<Value> {
    let result=(||->Result<Value>{
        if body.len()>128*1024 {bail!("Policy request exceeds 128 KiB");}
        let value=crate::strict_json::parse(std::str::from_utf8(&body)?)?;
        let request:Create=serde_json::from_value(value)?;
        let manifest=Manifest::create(request.rules)?;
        Ok(json!({"policy":Store::open(&app.data)?.put(&manifest)?,"provider_calls":0}))
    })();
    result.map(axum::Json).map_err(|error|crate::error(axum::http::StatusCode::BAD_REQUEST,error))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Check { text:String,stage:String,action:String }
pub(crate) async fn check_api(axum::extract::State(app):axum::extract::State<crate::App>,axum::extract::Path(hash):axum::extract::Path<String>,body:axum::body::Bytes)->crate::ApiResult<Value> {
    let result=(||->Result<Value>{
        if body.len()>512*1024 {bail!("Guardrail request exceeds 512 KiB");}
        let value=crate::strict_json::parse(std::str::from_utf8(&body)?)?;
        let request:Check=serde_json::from_value(value)?;
        Store::open(&app.data)?.get(&hash)?.check(&request.text,&request.stage,&request.action)
    })();
    result.map(axum::Json).map_err(|error|crate::error(axum::http::StatusCode::BAD_REQUEST,error))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Selection {
    pub input_policy_sha256:String,
    pub output_policy_sha256:String,
    pub action:String,
}
#[derive(Clone, Debug)]
pub(crate) struct Prepared {
    input:Manifest,
    output:Manifest,
    action:String,
}
impl Selection {
    pub fn prepare(&self,store:&Store)->Result<Prepared> {
        if !["observe","block"].contains(&self.action.as_str()) {bail!("Choose explicit guardrail action");}
        Ok(Prepared{input:store.get(&self.input_policy_sha256)?,output:store.get(&self.output_policy_sha256)?,action:self.action.clone()})
    }
}
impl Prepared {
    pub fn input(&self,text:&str)->Result<Value> {self.input.check(text,"input",&self.action)}
    pub fn output(&self,text:&str)->Result<Value> {self.output.check(text,"output",&self.action)}
    pub fn buffers_output(&self)->bool {self.action=="block"}
}
