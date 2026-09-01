//! RAII command and encoder lifecycle for Metal execution paths.
//!
//! A compute encoder is ended from `Drop`, including early-return paths.
//! Consuming [`CommandScope::commit`] makes submission explicit and prevents
//! accidental reuse of the command scope after commit.

#[cfg(target_os = "macos")]
use metal::{CommandBufferRef, ComputeCommandEncoderRef};

#[cfg(target_os = "macos")]
#[must_use = "a command scope must be explicitly committed or intentionally dropped"]
pub struct CommandScope<'a> {
    command: &'a CommandBufferRef,
}

#[cfg(target_os = "macos")]
impl<'a> CommandScope<'a> {
    pub fn new(command: &'a CommandBufferRef) -> Self {
        Self { command }
    }

    pub fn compute_encoder(&mut self) -> ComputeEncoderScope<'_> {
        ComputeEncoderScope {
            encoder: self.command.new_compute_command_encoder(),
            ended: false,
        }
    }

    pub fn raw(&self) -> &CommandBufferRef {
        self.command
    }

    pub fn commit(self) {
        self.command.commit();
    }
}

#[cfg(target_os = "macos")]
pub struct ComputeEncoderScope<'a> {
    encoder: &'a ComputeCommandEncoderRef,
    ended: bool,
}

#[cfg(target_os = "macos")]
impl ComputeEncoderScope<'_> {
    pub fn raw(&self) -> &ComputeCommandEncoderRef {
        self.encoder
    }

    pub fn end(mut self) {
        self.finish();
    }

    fn finish(&mut self) {
        if !self.ended {
            self.encoder.end_encoding();
            self.ended = true;
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for ComputeEncoderScope<'_> {
    fn drop(&mut self) {
        self.finish();
    }
}
