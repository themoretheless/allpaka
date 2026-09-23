use anyhow::{bail, Context, Result};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

pub const AVAILABLE: bool = cfg!(any(target_os = "macos", target_os = "windows"));
pub trait Vault {
    fn get(&self, service: &str, account: &str) -> Result<Option<String>>;
    fn set(&self, service: &str, account: &str, secret: &str) -> Result<()>;
    fn delete(&self, service: &str, account: &str) -> Result<()>;
}
pub struct SystemVault;
#[cfg(any(target_os = "macos", target_os = "windows"))]
impl Vault for SystemVault {
    fn get(&self, service: &str, account: &str) -> Result<Option<String>> {
        let entry = keyring::Entry::new(service, account)
            .map_err(|_| anyhow::anyhow!("Cannot access system credential store"))?;
        match entry.get_password() {
            Ok(value) => Ok(Some(value)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(_) => bail!("System credential store is locked or unavailable"),
        }
    }
    fn set(&self, service: &str, account: &str, secret: &str) -> Result<()> {
        keyring::Entry::new(service, account)
            .and_then(|e| e.set_password(secret))
            .map_err(|_| anyhow::anyhow!("Could not save key in system credential store"))
    }
    fn delete(&self, service: &str, account: &str) -> Result<()> {
        let entry = keyring::Entry::new(service, account)
            .map_err(|_| anyhow::anyhow!("Cannot access system credential store"))?;
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(_) => bail!("Could not remove saved key from system credential store"),
        }
    }
}
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
impl Vault for SystemVault {
    fn get(&self, _: &str, _: &str) -> Result<Option<String>> {
        bail!("System credential storage is not supported on this platform")
    }
    fn set(&self, _: &str, _: &str, _: &str) -> Result<()> {
        bail!("System credential storage is not supported on this platform")
    }
    fn delete(&self, _: &str, _: &str) -> Result<()> {
        bail!("System credential storage is not supported on this platform")
    }
}
pub struct Store {
    path: PathBuf,
    service: String,
}
impl Store {
    pub fn new(data: &Path) -> Self {
        Self {
            path: data.join("credentials.state"),
            service: format!("allpaka.studio:{}", data.display()),
        }
    }
    pub fn saved(&self) -> Result<BTreeSet<String>> {
        if !self.path.exists() {
            return Ok(BTreeSet::new());
        }
        let bytes = std::fs::read(&self.path)?;
        if bytes.len() > 16384 {
            bail!("Invalid credential index");
        }
        serde_json::from_slice(&bytes).context("Invalid credential index")
    }
    fn index(&self, ids: &BTreeSet<String>) -> Result<()> {
        let temp = self.path.with_extension("tmp");
        std::fs::write(&temp, serde_json::to_vec(ids)?)?;
        std::fs::rename(temp, &self.path)?;
        Ok(())
    }
    pub fn load(&self, vault: &dyn Vault, id: &str) -> Result<Option<String>> {
        if !self.saved()?.contains(id) {
            return Ok(None);
        }
        vault.get(&self.service, id)
    }
    pub fn update(&self, vault: &dyn Vault, id: &str, key: &str, persist: bool) -> Result<()> {
        let mut ids = self.saved()?;
        if persist && !key.is_empty() {
            // Register first: a failed vault operation may leave a harmless marker,
            // but a successfully stored key will always be discoverable for deletion.
            ids.insert(id.into());
            self.index(&ids)?;
            vault.set(&self.service, id, key)?;
        } else if ids.contains(id) {
            vault.delete(&self.service, id)?;
            ids.remove(id);
            self.index(&ids)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashMap, sync::Mutex};
    #[derive(Default)]
    struct FakeVault(Mutex<HashMap<(String, String), String>>);
    impl Vault for FakeVault {
        fn get(&self, s: &str, a: &str) -> Result<Option<String>> {
            Ok(self.0.lock().unwrap().get(&(s.into(), a.into())).cloned())
        }
        fn set(&self, s: &str, a: &str, k: &str) -> Result<()> {
            self.0
                .lock()
                .unwrap()
                .insert((s.into(), a.into()), k.into());
            Ok(())
        }
        fn delete(&self, s: &str, a: &str) -> Result<()> {
            self.0.lock().unwrap().remove(&(s.into(), a.into()));
            Ok(())
        }
    }
    fn temporary() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "studio-credentials-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&p).unwrap();
        p.canonicalize().unwrap()
    }
    #[test]
    fn secrets_are_outside_history_and_survive_store_recreation() {
        let path = temporary();
        let vault = FakeVault::default();
        let store = Store::new(&path);
        store.update(&vault, "test", "dummy-secret", true).unwrap();
        assert_eq!(
            Store::new(&path).load(&vault, "test").unwrap().as_deref(),
            Some("dummy-secret")
        );
        assert_eq!(
            std::fs::read_to_string(path.join("credentials.state")).unwrap(),
            "[\"test\"]"
        );
        store.update(&vault, "test", "replacement", true).unwrap();
        assert_eq!(
            store.load(&vault, "test").unwrap().as_deref(),
            Some("replacement")
        );
        store.update(&vault, "test", "memory-only", false).unwrap();
        assert!(store.load(&vault, "test").unwrap().is_none());
        assert!(vault.0.lock().unwrap().is_empty());
        store.update(&vault, "test", "", false).unwrap();
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn unavailable_vault_never_falls_back_to_plaintext() {
        struct Unavailable;
        impl Vault for Unavailable {
            fn get(&self, _: &str, _: &str) -> Result<Option<String>> {
                bail!("locked")
            }
            fn set(&self, _: &str, _: &str, _: &str) -> Result<()> {
                bail!("locked")
            }
            fn delete(&self, _: &str, _: &str) -> Result<()> {
                bail!("locked")
            }
        }
        let path = temporary();
        let store = Store::new(&path);
        assert!(store
            .update(&Unavailable, "test", "never-on-disk", true)
            .is_err());
        assert!(!std::fs::read_to_string(path.join("credentials.state"))
            .unwrap()
            .contains("never-on-disk"));
        assert!(store.load(&Unavailable, "test").is_err());
        assert!(store.update(&Unavailable, "test", "", false).is_err());
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn stores_are_scoped_to_the_history_directory() {
        let a = temporary();
        let b = temporary();
        let vault = FakeVault::default();
        Store::new(&a)
            .update(&vault, "test", "first", true)
            .unwrap();
        Store::new(&b)
            .update(&vault, "test", "second", true)
            .unwrap();
        assert_eq!(
            Store::new(&a).load(&vault, "test").unwrap().as_deref(),
            Some("first")
        );
        assert_eq!(
            Store::new(&b).load(&vault, "test").unwrap().as_deref(),
            Some("second")
        );
        std::fs::remove_dir_all(a).unwrap();
        std::fs::remove_dir_all(b).unwrap();
    }
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    #[test]
    #[ignore = "Writes and removes one dummy credential in the native OS store"]
    fn native_vault_roundtrip() {
        let path = temporary();
        let store = Store::new(&path);
        struct Cleanup<'a>(&'a Store);
        impl Drop for Cleanup<'_> {
            fn drop(&mut self) {
                let _ = self.0.update(&SystemVault, "test-only", "", false);
            }
        }
        let cleanup = Cleanup(&store);
        store
            .update(
                &SystemVault,
                "test-only",
                "allpaka-test-not-a-real-api-key",
                true,
            )
            .unwrap();
        assert_eq!(
            Store::new(&path)
                .load(&SystemVault, "test-only")
                .unwrap()
                .as_deref(),
            Some("allpaka-test-not-a-real-api-key")
        );
        store.update(&SystemVault, "test-only", "", false).unwrap();
        assert!(store.load(&SystemVault, "test-only").unwrap().is_none());
        drop(cleanup);
        std::fs::remove_dir_all(path).unwrap();
    }
}
