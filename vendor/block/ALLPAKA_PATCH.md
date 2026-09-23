Vendored from crates.io block 0.1.6 (MIT), used transitively by metal.

The opaque Class marker is an inhabited repr(C) struct instead of an empty enum.
Only its address is used for the Objective-C block isa pointer; its size is never
used to allocate or read the external object. This fixes uninhabited_static
without changing block layout or suppressing the compiler diagnostic.

Removed the unpublished objc_test_utils dev dependency from the packaged manifest.
Explicitly named the C ABI in previously implicit extern declarations.
The upstream README and remaining library source are unchanged. The upstream
package has no separate license file; its MIT declaration and author metadata
are preserved in Cargo.toml.
