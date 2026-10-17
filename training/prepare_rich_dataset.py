import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

def fetch_json(url, retries=3, timeout=20):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                print(f"Failed fetching {url}: {e}", file=sys.stderr)
    return None

def fetch_gsm8k(limit=150):
    print(f">> [1/6] GSM8K (limit: {limit})...")
    items = []
    for fpath in ["gsm8k_train.txt", "gsm8k_train_50.txt"]:
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                splits = [x.strip() for x in content.split("\n---\n") if x.strip()]
                for s in splits:
                    if s not in items:
                        items.append(s)
            if len(items) >= limit:
                return items[:limit]

    for offset in range(0, limit, 100):
        chunk_limit = min(100, limit - len(items))
        if chunk_limit <= 0:
            break
        url = f"https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k&config=main&split=train&offset={offset}&limit={chunk_limit}"
        data = fetch_json(url)
        if not data or "rows" not in data:
            break
        for r in data["rows"]:
            row = r["row"]
            q = row.get("question", "").strip()
            a = row.get("answer", "").strip()
            if q and a:
                items.append(f"Question: {q}\nAnswer:\n{a}")
        time.sleep(0.3)

    return items

def fetch_swe_bench(limit=150):
    print(f">> [2/6] SWE-bench Lite (limit: {limit})...")
    items = []
    for offset in range(100, 100 + limit, 100):
        chunk_limit = min(100, (100 + limit) - offset)
        url = f"https://datasets-server.huggingface.co/rows?dataset=princeton-nlp/SWE-bench_Lite&config=default&split=test&offset={offset}&limit={chunk_limit}"
        data = fetch_json(url)
        if not data or "rows" not in data:
            break
        for r in data["rows"]:
            row = r["row"]
            repo = row.get("repo", "")
            issue = row.get("problem_statement", "").strip()
            patch = row.get("patch", "").strip()
            if issue and patch:
                items.append(f"Repository: {repo}\nIssue Description:\n{issue}\n\nSolution Patch:\n{patch}")
        time.sleep(0.3)

    return items

def fetch_numina_math(limit=250):
    print(f">> [3/6] NuminaMath-CoT (limit: {limit})...")
    items = []
    for offset in range(0, limit, 100):
        chunk_limit = min(100, limit - len(items))
        if chunk_limit <= 0:
            break
        url = f"https://datasets-server.huggingface.co/rows?dataset=AI-MO/NuminaMath-CoT&config=default&split=train&offset={offset}&limit={chunk_limit}"
        data = fetch_json(url)
        if not data or "rows" not in data:
            break
        for r in data["rows"]:
            row = r["row"]
            src = row.get("source", "olympiad")
            prob = row.get("problem", "").strip()
            sol = row.get("solution", "").strip()
            if prob and sol:
                items.append(f"Source: {src}\nProblem: {prob}\n\nDetailed Solution:\n{sol}")
        time.sleep(0.3)

    return items

def fetch_math_500(limit=150):
    print(f">> [4/6] MATH-500 (limit: {limit})...")
    items = []
    for offset in range(50, 50 + limit, 100):
        chunk_limit = min(100, (50 + limit) - offset)
        url = f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500&config=default&split=test&offset={offset}&limit={chunk_limit}"
        data = fetch_json(url)
        if not data or "rows" not in data:
            break
        for r in data["rows"]:
            row = r["row"]
            subj = row.get("subject", "Math")
            prob = row.get("problem", "").strip()
            sol = row.get("solution", "").strip()
            if prob and sol:
                items.append(f"Subject: {subj}\nProblem: {prob}\n\nSolution:\n{sol}")
        time.sleep(0.3)

    return items

def get_systems_and_architecture_dataset():
    print(">> [5/6] Systems Architecture & Low-Level Engineering...")
    problems = [
        ("Architecture Challenge: Lock-Free MPSC Queue in Rust",
         """Design a lock-free Multi-Producer Single-Consumer (MPSC) bounded queue in Rust without using std::sync::mpsc or mutexes.
Requirements:
1. Support concurrent push() from multiple producer threads without locking.
2. Single consumer pop() in FIFO order.
3. Cache-padded atomic indices to avoid false sharing.
4. Correct memory ordering (Acquire, Release, Relaxed).

Detailed Solution:
```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

#[repr(align(64))]
struct CachePadded<T>(T);

pub struct LockFreeMPSC<T, const CAP: usize> {
    buffer: Box<[UnsafeCell<MaybeUninit<T>>; CAP]>,
    flags: Box<[AtomicUsize; CAP]>,
    head: CachePadded<AtomicUsize>,
    tail: CachePadded<AtomicUsize>,
}

unsafe impl<T: Send, const CAP: usize> Send for LockFreeMPSC<T, CAP> {}
unsafe impl<T: Send, const CAP: usize> Sync for LockFreeMPSC<T, CAP> {}

impl<T, const CAP: usize> LockFreeMPSC<T, CAP> {
    pub fn new() -> Self {
        assert!(CAP.is_power_of_two(), "Capacity must be power of two");
        let buffer = Box::new(unsafe { MaybeUninit::uninit().assume_init() });
        let mut flags = Vec::with_capacity(CAP);
        for _ in 0..CAP {
            flags.push(AtomicUsize::new(0));
        }
        Self {
            buffer,
            flags: flags.into_boxed_slice().try_into().unwrap_or_else(|_| panic!()),
            head: CachePadded(AtomicUsize::new(0)),
            tail: CachePadded(AtomicUsize::new(0)),
        }
    }

    pub fn push(&self, value: T) -> Result<(), T> {
        let mask = CAP - 1;
        loop {
            let h = self.head.0.load(Ordering::Relaxed);
            let t = self.tail.0.load(Ordering::Acquire);
            if h.wrapping_sub(t) >= CAP {
                return Err(value);
            }
            if self.head.0.compare_exchange_weak(h, h.wrapping_add(1), Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                let slot = h & mask;
                while self.flags[slot].compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_err() {
                    std::hint::spin_loop();
                }
                unsafe { (*self.buffer[slot].get()).write(value); }
                self.flags[slot].store(2, Ordering::Release);
                return Ok(());
            }
            std::hint::spin_loop();
        }
    }

    pub fn pop(&self) -> Option<T> {
        let mask = CAP - 1;
        let t = self.tail.0.load(Ordering::Relaxed);
        let h = self.head.0.load(Ordering::Acquire);
        if t == h {
            return None;
        }
        let slot = t & mask;
        if self.flags[slot].load(Ordering::Acquire) != 2 {
            return None;
        }
        let val = unsafe { (*self.buffer[slot].get()).assume_init_read() };
        self.flags[slot].store(0, Ordering::Release);
        self.tail.0.store(t.wrapping_add(1), Ordering::Release);
        Some(val)
    }
}
```"""),
        ("Systems Architecture: Bump-Pointer Monolithic Arena in C++20",
         """Design a zero-fragmentation, cache-aligned bump-pointer memory arena in C++20 for high-frequency game engine loops.
Requirements:
1. Monolithic 64-byte aligned allocation block.
2. Constant O(1) allocation with pointer bumping.
3. Constant O(1) instantaneous frame reset without individual deallocations.
4. Placement new forwarding for object construction.

Detailed Solution:
```cpp
#include <cstddef>
#include <new>
#include <utility>
#include <cstdint>

class FrameArena {
private:
    std::byte* buffer;
    size_t capacity;
    size_t offset;

    static size_t align_forward(size_t ptr, size_t alignment) {
        return (ptr + alignment - 1) & ~(alignment - 1);
    }

public:
    explicit FrameArena(size_t cap)
        : capacity(cap), offset(0) {
        buffer = static_cast<std::byte*>(::operator new(capacity, std::align_val_t{64}));
    }

    ~FrameArena() {
        ::operator delete(buffer, std::align_val_t{64});
    }

    FrameArena(const FrameArena&) = delete;
    FrameArena& operator=(const FrameArena&) = delete;

    template<typename T, typename... Args>
    T* allocate(Args&&... args) {
        size_t aligned_offset = align_forward(offset, alignof(T));
        size_t next_offset = aligned_offset + sizeof(T);
        if (next_offset > capacity) {
            return nullptr;
        }
        offset = next_offset;
        T* ptr = reinterpret_cast<T*>(buffer + aligned_offset);
        return new (ptr) T(std::forward<Args>(args)...);
    }

    void reset() {
        offset = 0;
    }
};
```""")
    ]

    items = []
    for title, body in problems:
        items.append(f"Title: {title}\n\n{body}")
    return items

def build_dataset(output_path="rich_train_dataset.txt"):
    all_items = []
    all_items.extend(fetch_gsm8k(limit=150))
    all_items.extend(fetch_swe_bench(limit=150))
    all_items.extend(fetch_numina_math(limit=250))
    all_items.extend(fetch_math_500(limit=150))
    all_items.extend(get_systems_and_architecture_dataset())

    random.seed(42)
    random.shuffle(all_items)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n---\n".join(all_items))

    print(f"Dataset saved to: {output_path}")
    print(f"Total problems: {len(all_items)}")
    print(f"File size: {os.path.getsize(output_path) / (1024.0 * 1024.0):.2f} MB")

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "rich_train_dataset.txt"
    build_dataset(out_file)
