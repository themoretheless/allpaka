//! RAII command and encoder lifecycle for Metal execution paths.
//!
//! A compute encoder is ended from `Drop`, including early-return paths.
//! Consuming [`CommandScope::commit`] makes submission explicit and prevents
//! accidental reuse of the command scope after commit.

#[cfg(target_os = "macos")]
use metal::{
    BlitCommandEncoderRef, CommandBufferRef, ComputeCommandEncoderRef,
    ComputePassDescriptorRef, MTLDispatchType,
};

#[cfg(target_os = "macos")]
use std::ops::Deref;

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

    pub fn compute_encoder(&self) -> ComputeEncoderScope<'a> {
        ComputeEncoderScope {
            encoder: self.command.new_compute_command_encoder(),
            ended: false,
        }
    }

    pub fn compute_command_encoder_with_descriptor(
        &self,
        descriptor: &ComputePassDescriptorRef,
    ) -> ComputeEncoderScope<'a> {
        ComputeEncoderScope {
            encoder: self
                .command
                .compute_command_encoder_with_descriptor(descriptor),
            ended: false,
        }
    }

    pub fn compute_command_encoder_with_dispatch_type(
        &self,
        dispatch_type: MTLDispatchType,
    ) -> ComputeEncoderScope<'a> {
        ComputeEncoderScope {
            encoder: self
                .command
                .compute_command_encoder_with_dispatch_type(dispatch_type),
            ended: false,
        }
    }

    pub fn new_blit_command_encoder(&self) -> BlitEncoderScope<'a> {
        BlitEncoderScope {
            encoder: self.command.new_blit_command_encoder(),
            ended: false,
        }
    }

    pub fn raw(&self) -> &CommandBufferRef {
        self.command
    }

    pub fn commit(self) {
        self.command.commit();
    }

    pub fn commit_and_wait(self) -> &'a CommandBufferRef {
        self.command.commit();
        self.command.wait_until_completed();
        self.command
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

    pub fn end_encoding(mut self) {
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
impl Deref for ComputeEncoderScope<'_> {
    type Target = ComputeCommandEncoderRef;

    fn deref(&self) -> &Self::Target {
        self.encoder
    }
}

#[cfg(target_os = "macos")]
impl Drop for ComputeEncoderScope<'_> {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(target_os = "macos")]
pub struct BlitEncoderScope<'a> {
    encoder: &'a BlitCommandEncoderRef,
    ended: bool,
}

#[cfg(target_os = "macos")]
impl BlitEncoderScope<'_> {
    pub fn raw(&self) -> &BlitCommandEncoderRef {
        self.encoder
    }

    pub fn end_encoding(mut self) {
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
impl Deref for BlitEncoderScope<'_> {
    type Target = BlitCommandEncoderRef;

    fn deref(&self) -> &Self::Target {
        self.encoder
    }
}

#[cfg(target_os = "macos")]
impl Drop for BlitEncoderScope<'_> {
    fn drop(&mut self) {
        self.finish();
    }
}
