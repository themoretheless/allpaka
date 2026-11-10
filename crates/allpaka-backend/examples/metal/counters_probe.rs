// Probe: which Metal counter sets does this GPU expose?
use metal::*;
use objc::runtime::Object;
use objc::{msg_send, sel, sel_impl};

fn main() {
    let device = Device::system_default().unwrap();
    println!("device: {}", device.name());
    probe_points(&device);
    for set in device.counter_sets() {
        println!("counter set: {:?}", set.name());
        unsafe {
            let counters: *const Object = msg_send![&*set, counters];
            let n: u64 = msg_send![counters, count];
            for i in 0..n {
                let c: *const Object = msg_send![counters, objectAtIndex: i];
                let name: *const Object = msg_send![c, name];
                let utf8: *const std::os::raw::c_char = msg_send![name, UTF8String];
                let name = std::ffi::CStr::from_ptr(utf8).to_string_lossy();
                println!("    counter: {name}");
            }
        }
    }
}

// supportsCounterSampling per sampling point (0=atCommandBoundary,
// 1=atDrawBoundary, 2=atDispatchBoundary, 3=atStageBoundary? guess)
fn probe_points(device: &Device) {
    for p in 0u64..6u64 {
        let ok: bool = unsafe { msg_send![device.as_ref(), supportsCounterSampling: p] };
        println!("sampling point {p}: {ok}");
    }
}
