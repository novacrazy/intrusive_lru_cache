Intrusive LRU Cache
===================

[![crates.io](https://img.shields.io/crates/v/intrusive-lru-cache.svg)](https://crates.io/crates/intrusive-lru-cache)
[![Documentation](https://docs.rs/intrusive-lru-cache/badge.svg)](https://docs.rs/intrusive-lru-cache)
[![MIT/Apache-2 licensed](https://img.shields.io/crates/l/intrusive-lru-cache.svg)](./LICENSE-Apache)
[![Build Status](https://github.com/novacrazy/intrusive_lru_cache/actions/workflows/CI.yml/badge.svg)](https://github.com/novacrazy/intrusive_lru_cache/actions/workflows/CI.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/novacrazy/7270c68e5927fa2a1ef2de9f4286010b/raw/intrusive-lru-cache-coverage.json)](https://github.com/novacrazy/intrusive_lru_cache/actions/workflows/CI.yml)
![Rust Version](https://img.shields.io/badge/rustc-1.84+-blue.svg)

This crate provides an LRU Cache implementation that is based on combining an intrusive doubly linked list and an intrusive red-black tree,
in the same node. Both data structures share the same allocations, which makes it quite efficient for a linked structure.

The [`LRUCache`] structure itself is not intrusive, and works like a regular cache. The intrusive part of the crate name is due to the
intrusive structures used internally.

This crate is `#![no_std]`, but requires [`alloc`](https://doc.rust-lang.org/alloc/) for the per-entry allocations.

# Example

```rust
use intrusive_lru_cache::LRUCache;

let mut lru: LRUCache<&'static str, &'static str> = LRUCache::default();

lru.insert("a", "1");
lru.insert("b", "2");
lru.insert("c", "3");

let _ = lru.get("b"); // updates LRU order

assert_eq!(lru.pop(), Some(("a", "1")));
assert_eq!(lru.pop(), Some(("c", "3")));
assert_eq!(lru.pop(), Some(("b", "2")));
assert_eq!(lru.pop(), None);
```

# Smart Entries

The `smart_*` methods return a [`SmartEntry`], which only updates the LRU order when the value is
accessed _mutably_. Immutable access (via [`SmartEntry::peek`] or `Deref`) leaves the order untouched,
while mutable access (via [`SmartEntry::get`] or `DerefMut`) bumps the entry to most-recently-used.
This lets a single lookup decide, after the fact, whether it counts as a "use".

```rust
use intrusive_lru_cache::LRUCache;

let mut lru: LRUCache<&'static str, &'static str> = LRUCache::default();

lru.insert("a", "1");
lru.insert("b", "2");

let mut entry = lru.smart_get("a").unwrap();

assert_eq!(*entry, "1"); // immutable access (Deref) does not bump the LRU order
assert!(!entry.is_most_recent());

*entry = "10"; // mutable access (DerefMut) bumps "a" to most-recently-used
assert!(entry.is_most_recent());
```

# Cargo Features
- `atomic` (default): Enables atomic links within the intrusive structures, making it thread-safe if
  `K` and `V` are `Send`/`Sync`. If you disable this feature, you can still use the cache in a single-threaded context.

# Minimum Supported Rust Version (MSRV)

This crate requires **Rust 1.84** or newer (it relies on the stabilized exposed-provenance pointer APIs).