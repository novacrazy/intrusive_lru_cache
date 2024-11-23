Intrusive LRU Cache
===================

[![crates.io](https://img.shields.io/crates/v/intrusive-lru-cache.svg)](https://crates.io/crates/intrusive-lru-cache)
[![Documentation](https://docs.rs/intrusive-lru-cache/badge.svg)](https://docs.rs/intrusive-lru-cache)
[![MIT/Apache-2 licensed](https://img.shields.io/crates/l/intrusive-lru-cache.svg)](./LICENSE-Apache)

This crate provides an LRU Cache implementation that is based on an intrusive doubly linked list and an intrusive red-black tree,
in the same node. Both data structures share the same allocations, which makes it very efficient for a linked structure.