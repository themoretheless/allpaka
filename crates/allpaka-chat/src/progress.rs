//! Bounded consecutive identical tool failures within one agent turn.
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
#[derive(Default)]
pub(crate) struct FailureLoop { previous:Option<[u8;32]>, count:usize }
impl FailureLoop {
    pub(crate) fn reset(&mut self){*self=Self::default();}
    pub(crate) fn observe(&mut self,call:&Value,error:Option<&str>)->bool{
        let Some(error)=error else{self.reset();return false;};
        let arguments=call["function"]["arguments"].as_str().unwrap_or("");
        let arguments=serde_json::from_str::<Value>(arguments).unwrap_or_else(|_|json!(arguments));
        let signature:[u8;32]=Sha256::digest(serde_json::to_vec(&json!([call["function"]["name"],arguments,error])).unwrap()).into();
        self.count=if self.previous==Some(signature){self.count+1}else{1};self.previous=Some(signature);
        self.count>=3
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn equivalent_arguments_repeat_but_success_changed_error_and_steering_reset(){
        let call=json!({"function":{"name":"read_file","arguments":"{\"path\":\"missing\",\"offset\":0}"}});
        let reordered=json!({"function":{"name":"read_file","arguments":"{ \"offset\": 0, \"path\": \"missing\" }"}});
        let mut guard=FailureLoop::default();assert!(!guard.observe(&call,Some("missing")));assert!(!guard.observe(&reordered,Some("missing")));assert!(guard.observe(&call,Some("missing")));
        assert!(!guard.observe(&call,None));assert!(!guard.observe(&call,Some("missing")));assert!(!guard.observe(&call,Some("permission")));guard.reset();assert!(!guard.observe(&call,Some("permission")));
    }
}
