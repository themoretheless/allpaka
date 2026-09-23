use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Root {
    pub alias: String,
    pub path: PathBuf,
    #[serde(default)]
    pub writable: bool,
    #[serde(default)]
    pub repository: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Project {
    pub id: String,
    pub name: String,
    pub roots: Vec<Root>,
    #[serde(default)]
    pub instructions: String,
}
impl Project {
    pub fn validate(&mut self, history: &Path) -> Result<()> {
        if self.name.trim().is_empty()
            || self.name.len() > 100
            || self.roots.is_empty()
            || self.roots.len() > 20
            || self.instructions.len() > 16000
        {
            bail!("Project needs a name and 1–20 context folders");
        }
        let mut aliases = std::collections::HashSet::new();
        for r in &mut self.roots {
            if r.alias.is_empty()
                || r.alias.len() > 40
                || !r
                    .alias
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
                || !aliases.insert(r.alias.clone())
            {
                bail!("Context aliases must be unique letters, digits, - or _");
            }
            r.path = r
                .path
                .canonicalize()
                .with_context(|| format!("Folder unavailable: {}", r.path.display()))?;
            if !r.path.is_dir() {
                bail!("Context must be a folder or local repository");
            }
            if r.path == history || r.path.starts_with(history) {
                bail!("Conversation storage cannot be a context root");
            }
            r.repository = r.path.join(".git").exists();
        }
        Ok(())
    }
    pub fn root<'a>(&'a self, path: &str, write: bool) -> Result<(&'a Path, String)> {
        let (alias, relative) = path.split_once('/').unwrap_or((path, "."));
        let root = self
            .roots
            .iter()
            .find(|r| r.alias == alias)
            .context("Use context-alias/relative-path; list context roots with path '.'")?;
        if write && !root.writable {
            bail!("This context folder is read-only");
        }
        Ok((&root.path, relative.into()))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Image {
    pub name: String,
    pub mime: String,
    pub data: String,
}
pub fn validate_images(images: &[Image], provider: &str) -> Result<()> {
    if images.len() > 4 {
        bail!("Up to 4 images per message");
    }
    if provider == "deepseek" && !images.is_empty() {
        bail!("This DeepSeek connection is text-only; choose a vision model with another provider");
    }
    let mut size = 0;
    for i in images {
        if !["image/png", "image/jpeg", "image/webp", "image/gif"].contains(&i.mime.as_str())
            || i.name.len() > 250
            || i.data.is_empty()
            || i.data.len() % 4 != 0
            || !i
                .data
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"+/=".contains(&b))
        {
            bail!("Invalid image attachment");
        }
        size += i.data.len();
    }
    if size > 8_000_000 {
        bail!("Images exceed 6 MB total before base64 encoding");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn root_permissions_are_independent() {
        let p = Project {
            id: "test".into(),
            name: "test".into(),
            instructions: String::new(),
            roots: vec![
                Root {
                    alias: "source".into(),
                    path: PathBuf::from("/tmp/source"),
                    writable: true,
                    repository: true,
                },
                Root {
                    alias: "reference".into(),
                    path: PathBuf::from("/tmp/reference"),
                    writable: false,
                    repository: false,
                },
            ],
        };
        assert!(p.root("source/a", true).is_ok());
        assert!(p.root("reference/a", false).is_ok());
        assert!(p.root("reference/a", true).is_err());
        assert!(p.root("other/a", false).is_err());
    }
    #[test]
    fn text_provider_rejects_image_context() {
        let image = Image {
            name: "x.png".into(),
            mime: "image/png".into(),
            data: "aGVsbG8=".into(),
        };
        assert!(validate_images(&[image.clone()], "deepseek").is_err());
        assert!(validate_images(&[image], "anthropic").is_ok());
    }
}
