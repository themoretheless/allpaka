// window-probe.swift - does Metal *say* anything when an over-mapped
// bytesNoCopy weight window stops delivering bytes?
//
// Maps N independent overlapping window sets over the same file the way
// allpaka's add_mapping does (cap-sized windows stepping by cap - 2 GiB, each
// handed to newBufferWithBytesNoCopy), dispatches one scattered page-stride
// checksum kernel per window, and compares against the same bytes summed on
// the host from the mmap. Prints commandBuffer.error for every buffer, so
// "is a dead window loud or silent at the driver?" gets answered, not assumed.
//
// build:  swiftc -O window-probe.swift -o window-probe
// run:    ./window-probe <file> <capGiB> <copies>

import Foundation
import Metal

let SOURCE = """
#include <metal_stdlib>
using namespace metal;
// Page-aligned samples spread over the whole window: a window that reads back
// zeros is then distinguishable from one that merely misaligns.
kernel void window_sum(device const uchar *src [[buffer(0)]],
                       device uint *out [[buffer(1)]],
                       constant uint *meta [[buffer(2)]],
                       uint tid [[thread_position_in_grid]]) {
    uint pages = meta[0];
    uint window = meta[3];
    uint acc = 0u;
    for (uint s = 0; s < 256u; ++s) {
        uint page = (s * 251u + tid * 7u) % pages;
        acc += src[(uint64_t)page * 16384u];
    }
    out[window * 256u + tid] = acc;
}
"""

let args = CommandLine.arguments
guard args.count >= 4, let capGiB = Double(args[2]), let copies = Int(args[3]) else {
    print("usage: window-probe <file> <capGiB> <copies>")
    exit(2)
}

let path = args[1]
let pageSize = 16384
let threads = 256
let samples = 256
let cap = UInt64(capGiB * Double(1 << 30))
let overlap = UInt64(2) << 30
let step = cap > overlap ? cap - overlap : cap / 2

let fh = open(path, O_RDONLY)
guard fh >= 0 else { print("open failed"); exit(1) }
let size = UInt64(lseek(fh, 0, SEEK_END))
let base = mmap(nil, Int(size), PROT_READ, MAP_PRIVATE, fh, 0)
guard base != MAP_FAILED else { print("mmap failed"); exit(1) }
print("file size \(String(format: "%.2f", Double(size) / Double(1 << 30))) GiB, "
    + "cap \(capGiB) GiB, step \(String(format: "%.1f", Double(step) / Double(1 << 30))) GiB")

guard let device = MTLCreateSystemDefaultDevice() else { print("no Metal device"); exit(1) }
let fn = try! device.makeLibrary(source: SOURCE, options: nil).makeFunction(name: "window_sum")!
let pipeline = try! device.makeComputePipelineState(function: fn)
let queue = device.makeCommandQueue()!

var windows: [(start: UInt64, length: UInt64)] = []
var start: UInt64 = 0
while start < size {
    let length = min(cap, size - start) & ~UInt64(pageSize - 1)
    if length >= UInt64(pageSize) { windows.append((start, length)) }
    if start + cap >= size { break }
    start += step
}
let outBuf = device.makeBuffer(length: MemoryLayout<UInt32>.stride * threads * max(windows.count, 1),
                               options: [])!
let gpu = outBuf.contents().assumingMemoryBound(to: UInt32.self)
let host = base!.assumingMemoryBound(to: UInt8.self)

var totalSpan = 0.0
var declines = 0, errors = 0, badWindows = 0, uninformative = 0, residencyFails = 0
for copy in 0..<copies {
    var buffers: [(MTLBuffer, UInt64)] = []
    for w in windows {
        let p = base!.advanced(by: Int(w.start))
        if let b = device.makeBuffer(bytesNoCopy: p, length: Int(w.length), options: [], deallocator: nil) {
            buffers.append((b, w.start))
        } else {
            declines += 1
            print("  copy \(copy): no-copy DECLINED at \(String(format: "%.1f", Double(w.start) / Double(1 << 30))) GiB")
        }
    }
    totalSpan += Double(buffers.reduce(0) { $0 + $1.0.length }) / Double(1 << 30)
    print("copy \(copy): \(buffers.count) windows, mapped span now \(String(format: "%.1f", totalSpan)) GiB")

    // allpaka's ensure_residency: one MTLResidencySet on the queue, every
    // window added as an allocation, commit + requestResidency.
    if ProcessInfo.processInfo.environment["RESIDENT"] != nil {
        do {
            let desc = MTLResidencySetDescriptor()
            let set = try device.makeResidencySet(descriptor: desc)
            queue.addResidencySet(set)
            for item in buffers { set.addAllocation(item.0) }
            set.commit()
            try set.requestResidency()
            print("  residency requested for \(buffers.count) allocations")
        } catch {
            print("  RESIDENCY THREW: \(error)")
            residencyFails += 1
        }
    }

    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(outBuf, offset: 0, index: 1)
    for (i, item) in buffers.enumerated() {
        let pages = UInt32(item.0.length / pageSize)
        var meta: [UInt32] = [pages, UInt32(threads), UInt32(samples), UInt32(i)]
        let metaBuf = device.makeBuffer(bytes: &meta, length: MemoryLayout<UInt32>.stride * 4, options: [])!
        enc.setBuffer(metaBuf, offset: 0, index: 2)
        enc.setBuffer(item.0, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    if let err = cmd.error as NSError? {
        errors += 1
        print("  commandBuffer.error: \(err.domain) code \(err.code): \(err.localizedDescription)")
    }

    for (i, item) in buffers.enumerated() {
        let b = item.0
        let pages = UInt32(b.length / pageSize)
        var hostTotal: UInt64 = 0
        var zeros = 0
        for t in 0..<threads {
            var acc: UInt64 = 0
            for s in 0..<samples {
                let page = (UInt32(s) &* 251 &+ UInt32(t) &* 7) % pages
                let byte = UInt64(host[Int(item.1) + Int(page) * pageSize])
                acc &+= byte
                if byte == 0 { zeros += 1 }
            }
            hostTotal &+= acc
        }
        var gpuTotal: UInt64 = 0
        for t in 0..<threads { gpuTotal &+= UInt64(gpu[i * threads + t]) }
        if hostTotal == 0 {
            uninformative += 1
            continue
        }
        if gpuTotal != hostTotal {
            badWindows += 1
            print("  WINDOW \(i) at \(String(format: "%.1f", Double(b.length) / Double(1 << 30))) GiB: "
                + "gpu \(gpuTotal) != host \(hostTotal) (\(zeros)/\(threads * samples) sampled bytes zero on host)")
        }
    }
}
print("span \(String(format: "%.1f", totalSpan)) GiB over \(copies) copies: "
    + "declines \(declines), command-buffer errors \(errors), "
    + "residency failures \(residencyFails), "
    + "windows reading wrong \(badWindows), uninformative (all-zero host) \(uninformative)")
