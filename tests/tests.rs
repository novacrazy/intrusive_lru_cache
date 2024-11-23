use intrusive_lru_cache::LRUCache;

#[test]
fn test_unsync_lru() {
    let mut lru: LRUCache<String, String> = LRUCache::default();

    lru.insert("a".to_string(), "1".to_string());
    lru.insert("b".to_string(), "2".to_string());
    lru.insert("c".to_string(), "3".to_string());

    let _ = lru.get("b");

    assert_eq!(lru.pop(), Some(("a".to_string(), "1".to_string())));
    assert_eq!(lru.pop(), Some(("c".to_string(), "3".to_string())));
    assert_eq!(lru.pop(), Some(("b".to_string(), "2".to_string())));
    assert_eq!(lru.pop(), None);
}
