#[test]
fn test_lru_cache() {
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
}
