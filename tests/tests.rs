#![allow(clippy::drop_non_drop)]

#[test]
fn test_lru_cache() {
    use intrusive_lru_cache::LRUCache;
    let mut lru: LRUCache<&'static str, &'static str> = LRUCache::default();

    lru.insert("a", "1");
    lru.insert("b", "2");
    lru.insert("c", "3");

    let _ = lru.get("b"); // updates LRU order

    assert_eq!(*lru.smart_get("a").unwrap(), "1");

    assert_eq!(lru.pop(), Some(("a", "1")));
    assert_eq!(lru.pop(), Some(("c", "3")));
    assert_eq!(lru.pop(), Some(("b", "2")));
    assert_eq!(lru.pop(), None);

    lru.insert_or_get("a", "1");
    lru.insert_or_get("a", "1");

    lru.remove("a");

    let mut iter = lru.smart_iter();

    let a = iter.next();

    // it would be nice if the SmartEntry were limited by the iterator lifetime,
    // but that would require borrowing from the iterator, which is not possible
    drop(iter);
    drop(a);
}
