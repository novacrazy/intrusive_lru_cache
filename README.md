Intrusive LRU Cache
===================

[![crates.io](https://img.shields.io/crates/v/intrusive-lru-cache.svg)](https://crates.io/crates/intrusive-lru-cache)
[![Documentation](https://docs.rs/intrusive-lru-cache/badge.svg)](https://docs.rs/intrusive-lru-cache)
[![MIT/Apache-2 licensed](https://img.shields.io/crates/l/intrusive-lru-cache.svg)](./LICENSE-Apache)

This crate provides an LRU Cache implementation that is based on combining an intrusive doubly linked list and an intrusive red-black tree,
in the same node. Both data structures share the same allocations, which makes it quite efficient for a linked structure.

The [`LRUCache`] structure itself is not intrusive, and works like a regular cache. The intrusive part of the crate name is due to the
intrusive structures used internally.

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

## Cargo Features
- `atomic` (default): Enables atomic links within the intrusive structures, making it thread-safe if
    `K` and `V` are `Send`/`Sync`. If you disable this feature, you can still use the cache in a single-threaded context.