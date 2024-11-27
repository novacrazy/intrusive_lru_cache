#![allow(clippy::drop_non_drop)]

use intrusive_lru_cache::LRUCache;

fn make_lru() -> LRUCache<&'static str, &'static str> {
    let mut lru = LRUCache::default();
    lru.insert("a", "1");
    lru.insert("b", "2");
    lru.insert("c", "3");
    lru
}

#[test]
fn test_lru_cache() {
    let mut lru = make_lru();

    println!("{} bytes", <LRUCache<&str, &str>>::NODE_SIZE);
    println!("{} bytes", lru.memory_footprint());

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

#[test]
fn test_pop_highest() {
    let mut lru = make_lru();

    // c > b > a as char values
    assert!('c' > 'b' && 'b' > 'a');
    assert_eq!(lru.pop_highest(), Some(("c", "3")));
    assert_eq!(lru.pop_highest(), Some(("b", "2")));
    assert_eq!(lru.pop_highest(), Some(("a", "1")));
    assert_eq!(lru.pop_highest(), None);
}

#[test]
fn test_pop_lowest() {
    let mut lru = make_lru();

    // a < b < c as char values
    assert!('a' < 'b' && 'b' < 'c');
    assert_eq!(lru.pop_lowest(), Some(("a", "1")));
    assert_eq!(lru.pop_lowest(), Some(("b", "2")));
    assert_eq!(lru.pop_lowest(), Some(("c", "3")));
    assert_eq!(lru.pop_lowest(), None);
}

#[test]
fn test_retain() {
    let mut lru = make_lru();

    lru.retain(|&k, _| k == "a" || k == "b");

    assert_eq!(lru.pop(), Some(("a", "1")));
    assert_eq!(lru.pop(), Some(("b", "2")));
    assert_eq!(lru.pop(), None);
}

#[test]
fn test_get_or_insert2() {
    let mut lru = LRUCache::<String, String>::unbounded();

    let mut test = false;

    let _ = lru.get_or_insert2("test", || {
        test = true;
        String::from("test")
    });
}
