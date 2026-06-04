#![allow(clippy::drop_non_drop)]

use intrusive_lru_cache::{Bound, GetOrInsertResult, InsertOrGetResult, LRUCache};

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

    //for x in lru.smart_iter() {
    //    _ = x.remove();
    //}

    //let mut lru = make_lru();
    // let mut iter = lru.smart_iter();
    //_ = iter.next().unwrap().remove();
    //_ = iter.next_back().unwrap().remove();
    //_ = iter.next_back().unwrap().remove();
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

    _ = lru.smart_get("test").unwrap().remove();
}

#[test]
fn test_new_and_capacity() {
    let lru = LRUCache::<i32, i32>::new(10);
    assert_eq!(lru.capacity(), 10);
    assert!(lru.is_empty());
    assert!(!lru.is_full());

    let lru = LRUCache::<i32, i32>::unbounded();
    assert_eq!(lru.capacity(), usize::MAX);

    let lru: LRUCache<i32, i32> = LRUCache::default();
    assert_eq!(lru.capacity(), usize::MAX);
}

#[test]
fn test_len_is_empty_is_full() {
    let mut lru = LRUCache::<&str, &str>::new(2);
    assert!(lru.is_empty());
    assert_eq!(lru.len(), 0);

    lru.insert("a", "1");
    assert_eq!(lru.len(), 1);
    assert!(!lru.is_empty());
    assert!(!lru.is_full());

    lru.insert("b", "2");
    assert_eq!(lru.len(), 2);
    assert!(lru.is_full());
}

#[test]
fn test_memory_footprint() {
    let lru = make_lru();
    let expected = <LRUCache<&str, &str>>::NODE_SIZE * 3 + size_of::<LRUCache<&str, &str>>();
    assert_eq!(lru.memory_footprint(), expected);
}

#[test]
fn test_peek_and_get() {
    let mut lru = make_lru();

    // peek does not update LRU order
    assert_eq!(lru.peek("a"), Some(&"1"));
    assert_eq!(lru.peek("missing"), None);

    // a is still the oldest because peek didn't bump it
    assert_eq!(lru.peek_oldest(), Some((&"a", &"1")));
    assert_eq!(lru.peek_newest(), Some((&"c", &"3")));

    // get updates LRU order
    assert_eq!(lru.get("a"), Some(&mut "1"));
    assert_eq!(lru.peek_newest(), Some((&"a", &"1")));
    assert_eq!(lru.get("missing"), None);

    // get mutation
    *lru.get("b").unwrap() = "20";
    assert_eq!(lru.peek("b"), Some(&"20"));
}

#[test]
fn test_peek_newest_oldest_empty() {
    let mut lru = LRUCache::<i32, i32>::unbounded();
    assert_eq!(lru.peek_newest(), None);
    assert_eq!(lru.peek_oldest(), None);
    assert_eq!(lru.get_newest(), None);
}

#[test]
fn test_get_newest() {
    let mut lru = make_lru();
    let (k, v) = lru.get_newest().unwrap();
    assert_eq!(k, &"c");
    *v = "30";
    assert_eq!(lru.peek("c"), Some(&"30"));
}

#[test]
fn test_promote_demote() {
    let mut lru = make_lru();

    // promote a to the front
    lru.promote("a");
    assert_eq!(lru.peek_newest(), Some((&"a", &"1")));

    // demote a to the back
    lru.demote("a");
    assert_eq!(lru.peek_oldest(), Some((&"a", &"1")));

    // demote on the element already at the back: no-op path
    lru.demote("a");
    assert_eq!(lru.peek_oldest(), Some((&"a", &"1")));

    // promote on element already at front: no-op path
    lru.promote("c");
    lru.promote("c");
    assert_eq!(lru.peek_newest(), Some((&"c", &"3")));

    // missing keys are no-ops
    lru.promote("missing");
    lru.demote("missing");
}

#[test]
fn test_contains() {
    let lru = make_lru();
    assert!(lru.contains("a"));
    assert!(!lru.contains("missing"));
}

#[test]
fn test_insert_returns_old() {
    let mut lru = LRUCache::<&str, &str>::unbounded();
    assert_eq!(lru.insert("a", "1"), None);
    assert_eq!(lru.insert("a", "2"), Some("1"));
    assert_eq!(lru.peek("a"), Some(&"2"));
}

#[test]
fn test_insert_eviction() {
    let mut lru = LRUCache::<&str, &str>::new(2);
    lru.insert("a", "1");
    lru.insert("b", "2");
    lru.insert("c", "3"); // evicts a

    assert!(!lru.contains("a"));
    assert!(lru.contains("b"));
    assert!(lru.contains("c"));
    assert_eq!(lru.len(), 2);
}

#[test]
fn test_remove() {
    let mut lru = make_lru();
    assert_eq!(lru.remove("b"), Some("2"));
    assert_eq!(lru.remove("b"), None);
    assert_eq!(lru.len(), 2);
}

#[test]
fn test_get_or_insert() {
    let mut lru = LRUCache::<&str, i32>::unbounded();

    // inserted branch
    match lru.get_or_insert("a", || 1) {
        GetOrInsertResult::Inserted(v) => assert_eq!(*v, 1),
        _ => panic!("expected Inserted"),
    }

    // existed branch
    match lru.get_or_insert("a", || 99) {
        GetOrInsertResult::Existed(v, k) => {
            assert_eq!(*v, 1);
            assert_eq!(k, "a");
        }
        _ => panic!("expected Existed"),
    }

    // deref / deref_mut / into_inner
    let mut res = lru.get_or_insert("a", || 0);
    assert_eq!(*res, 1);
    *res = 5;
    assert_eq!(*res.into_inner(), 5);

    // deref / deref_mut on the freshly-inserted variant
    let mut res = lru.get_or_insert("c", || 2);
    assert_eq!(*res, 2);
    *res = 3;
    assert_eq!(*res.into_inner(), 3);
}

#[test]
fn test_insert_or_get() {
    let mut lru = LRUCache::<&str, i32>::unbounded();

    match lru.insert_or_get("a", 1) {
        InsertOrGetResult::Inserted(v) => assert_eq!(*v, 1),
        _ => panic!("expected Inserted"),
    }

    match lru.insert_or_get("a", 99) {
        InsertOrGetResult::Existed(v, k, val) => {
            assert_eq!(*v, 1);
            assert_eq!(k, "a");
            assert_eq!(val, 99);
        }
        _ => panic!("expected Existed"),
    }

    // deref / deref_mut / into_inner on both variants
    let mut res = lru.insert_or_get("a", 0);
    assert_eq!(*res, 1);
    *res = 7;
    assert_eq!(*res.into_inner(), 7);

    // deref / deref_mut on the freshly-inserted variant
    let mut res = lru.insert_or_get("c", 2);
    assert_eq!(*res, 2);
    *res = 3;
    assert_eq!(*res.into_inner(), 3);
}

#[test]
fn test_pop_empty() {
    let mut lru = LRUCache::<i32, i32>::unbounded();
    assert_eq!(lru.pop(), None);
    assert_eq!(lru.pop_highest(), None);
    assert_eq!(lru.pop_lowest(), None);
}

#[test]
fn test_resize_and_shrink() {
    let mut lru = make_lru();
    lru.resize(1);
    assert_eq!(lru.len(), 1);
    assert_eq!(lru.capacity(), 1);
    // only the most recently used remains
    assert!(lru.contains("c"));

    let mut lru = make_lru();
    lru.set_max_capacity(1);
    assert_eq!(lru.len(), 3); // set_max_capacity does not evict
    lru.shrink();
    assert_eq!(lru.len(), 1);
}

#[test]
fn test_shrink_by() {
    let mut lru = make_lru();
    lru.shrink_by(2);
    assert_eq!(lru.len(), 1);

    // shrink_by past the end breaks early
    lru.shrink_by(10);
    assert_eq!(lru.len(), 0);
}

#[test]
fn test_shrink_with() {
    let mut lru = make_lru();
    lru.set_max_capacity(1);

    let mut removed = Vec::new();
    lru.shrink_with(|k, v| removed.push((k, v)));
    assert_eq!(removed, vec![("a", "1"), ("b", "2")]);
    assert_eq!(lru.len(), 1);
}

#[test]
fn test_shrink_by_with() {
    let mut lru = make_lru();

    let mut removed = Vec::new();
    lru.shrink_by_with(2, |k, v| removed.push((k, v)));
    assert_eq!(removed, vec![("a", "1"), ("b", "2")]);
    assert_eq!(lru.len(), 1);

    // breaks early when exhausted
    let mut removed = Vec::new();
    lru.shrink_by_with(10, |k, v| removed.push((k, v)));
    assert_eq!(removed, vec![("c", "3")]);
    assert!(lru.is_empty());
}

#[test]
fn test_clear() {
    let mut lru = make_lru();
    lru.clear();
    assert!(lru.is_empty());
    assert_eq!(lru.pop(), None);
}

#[test]
fn test_keys_and_iters() {
    let mut lru = make_lru();
    lru.get("a"); // bump a to front, ord order unchanged

    let keys: Vec<_> = lru.keys().copied().collect();
    assert_eq!(keys, vec!["a", "b", "c"]);

    // ord order
    let ord: Vec<_> = lru.iter_peek_ord().map(|(k, v)| (*k, *v)).collect();
    assert_eq!(ord, vec![("a", "1"), ("b", "2"), ("c", "3")]);

    // lru order: most recent first (a was just bumped)
    let lru_order: Vec<_> = lru.iter_peek_lru().map(|(k, v)| (*k, *v)).collect();
    assert_eq!(lru_order, vec![("a", "1"), ("c", "3"), ("b", "2")]);

    // reversed
    let rev: Vec<_> = lru.iter_peek_lru().rev().map(|(k, v)| (*k, *v)).collect();
    assert_eq!(rev, vec![("b", "2"), ("c", "3"), ("a", "1")]);
}

#[test]
fn test_peek_range_and_range() {
    let mut lru: LRUCache<i32, i32> = (0..5).map(|i| (i, i * 10)).collect();

    let peeked: Vec<_> = lru.peek_range(&1, &4).map(|(k, v)| (*k, *v)).collect();
    assert_eq!(peeked, vec![(1, 10), (2, 20), (3, 30)]);

    // range bumps and allows mutation
    for (_, v) in lru.range(&1, &3) {
        *v += 1;
    }
    assert_eq!(lru.peek(&1), Some(&11));
    assert_eq!(lru.peek(&2), Some(&21));

    // 1 and 2 were bumped most recently
    assert_eq!(lru.peek_newest(), Some((&2, &21)));
}

#[test]
fn test_smart_get() {
    let mut lru = make_lru();

    // peek via smart entry does not bump
    let entry = lru.smart_get("a").unwrap();
    assert_eq!(entry.key(), &"a");
    assert_eq!(entry.peek(), (&"a", &"1"));
    assert_eq!(entry.peek_value(), &"1");
    assert_eq!(*entry, "1");
    assert!(!entry.is_most_recent());
    drop(entry);
    assert_eq!(lru.peek_oldest(), Some((&"a", &"1")));

    // get bumps
    let mut entry = lru.smart_get("a").unwrap();
    let (k, v) = entry.get();
    assert_eq!(k, &"a");
    *v = "10";
    assert!(entry.is_most_recent());
    drop(entry);
    assert_eq!(lru.peek_newest(), Some((&"a", &"10")));

    assert!(lru.smart_get("missing").is_none());
}

#[test]
fn test_smart_get_deref_mut() {
    let mut lru = make_lru();
    let mut entry = lru.smart_get("a").unwrap();
    *entry = "10"; // deref_mut bumps
    assert_eq!(*entry.get_value(), "10");
    drop(entry);
    assert_eq!(lru.peek_newest(), Some((&"a", &"10")));
}

#[test]
fn test_smart_get_into_mut_into_ref() {
    let mut lru = make_lru();

    let entry = lru.smart_get("a").unwrap();
    let (k, v) = entry.into_ref();
    assert_eq!((k, v), (&"a", &"1"));

    let mut lru = make_lru();
    let entry = lru.smart_get("a").unwrap();
    let (k, v) = entry.into_mut();
    assert_eq!(k, &"a");
    *v = "10";
    assert_eq!(lru.peek_newest(), Some((&"a", &"10")));
}

#[test]
fn test_smart_get_remove() {
    let mut lru = make_lru();
    let entry = lru.smart_get("b").unwrap();
    assert_eq!(entry.remove(), ("b", "2"));
    assert_eq!(lru.len(), 2);
    assert!(!lru.contains("b"));
}

#[test]
fn test_smart_get_oldest() {
    let mut lru = make_lru();
    let entry = lru.smart_get_oldest().unwrap();
    assert_eq!(entry.key(), &"a");
    drop(entry);

    let mut empty = LRUCache::<i32, i32>::unbounded();
    assert!(empty.smart_get_oldest().is_none());
}

#[test]
fn test_smart_range() {
    let mut lru: LRUCache<i32, i32> = (0..5).map(|i| (i, i * 10)).collect();

    let keys: Vec<_> = lru.smart_range(&1, &4).map(|e| *e.key()).collect();
    assert_eq!(keys, vec![1, 2, 3]);

    for mut entry in lru.smart_range(&1, &3) {
        *entry.get_value() += 1;
    }
    assert_eq!(lru.peek(&1), Some(&11));
    assert_eq!(lru.peek(&2), Some(&21));
}

#[test]
fn test_smart_bounds() {
    let mut lru: LRUCache<i32, i32> = (0..5).map(|i| (i, i * 10)).collect();

    let entry = lru.smart_upper_bound(Bound::Included(&3)).unwrap();
    assert_eq!(entry.key(), &3);
    drop(entry);

    let entry = lru.smart_upper_bound(Bound::Excluded(&3)).unwrap();
    assert_eq!(entry.key(), &2);
    drop(entry);

    let entry = lru.smart_lower_bound(Bound::Included(&3)).unwrap();
    assert_eq!(entry.key(), &3);
    drop(entry);

    let entry = lru.smart_lower_bound(Bound::Excluded(&3)).unwrap();
    assert_eq!(entry.key(), &4);
    drop(entry);

    let entry = lru.smart_lower_bound(Bound::Unbounded).unwrap();
    assert_eq!(entry.key(), &0);
    drop(entry);

    // out of range -> None
    assert!(lru.smart_upper_bound(Bound::Excluded(&0)).is_none());
    assert!(lru.smart_lower_bound(Bound::Excluded(&4)).is_none());
}

#[test]
fn test_smart_iter_no_remove() {
    let mut lru = make_lru();

    let keys: Vec<_> = lru.smart_iter().map(|e| *e.key()).collect();
    assert_eq!(keys, vec!["a", "b", "c"]);

    // mutate via smart_iter (bumps order)
    for mut entry in lru.smart_iter() {
        if *entry.key() == "b" {
            *entry.get_value() = "20";
        }
    }
    assert_eq!(lru.peek("b"), Some(&"20"));
    assert_eq!(lru.peek_newest(), Some((&"b", &"20")));

    // reversed
    let keys: Vec<_> = lru.smart_iter().rev().map(|e| *e.key()).collect();
    assert_eq!(keys, vec!["c", "b", "a"]);
}

#[test]
fn test_clone_preserves_order() {
    let mut lru = make_lru();
    lru.get("a"); // bump a

    let cloned = lru.clone();
    let order: Vec<_> = cloned.iter_peek_lru().map(|(k, v)| (*k, *v)).collect();
    let orig: Vec<_> = lru.iter_peek_lru().map(|(k, v)| (*k, *v)).collect();
    assert_eq!(order, orig);
    assert_eq!(cloned.capacity(), lru.capacity());
}

#[test]
fn test_debug() {
    let lru = make_lru();
    let s = format!("{lru:?}");
    assert!(s.contains("\"a\""));
    assert!(s.contains("\"1\""));
}

#[test]
fn test_index() {
    let lru = make_lru();
    assert_eq!(lru["a"], "1");
}

#[test]
#[should_panic(expected = "no entry found for key")]
fn test_index_panics() {
    let lru = make_lru();
    let _ = lru["missing"];
}

#[test]
fn test_extend() {
    let mut lru = LRUCache::<&str, &str>::unbounded();
    lru.extend([("a", "1"), ("b", "2")]);
    assert_eq!(lru.len(), 2);
    assert_eq!(lru.peek("a"), Some(&"1"));
}

#[test]
fn test_into_iter() {
    let lru = make_lru();
    // most recently used first
    let items: Vec<_> = lru.into_iter().collect();
    assert_eq!(items, vec![("c", "3"), ("b", "2"), ("a", "1")]);
}

#[test]
fn test_into_iter_rev() {
    let lru = make_lru();
    let items: Vec<_> = lru.into_iter().rev().collect();
    assert_eq!(items, vec![("a", "1"), ("b", "2"), ("c", "3")]);
}

#[test]
fn test_into_iter_double_ended() {
    let lru = make_lru();
    let mut iter = lru.into_iter();
    assert_eq!(iter.next(), Some(("c", "3")));
    assert_eq!(iter.next_back(), Some(("a", "1")));
    assert_eq!(iter.next(), Some(("b", "2")));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next_back(), None);
}

#[test]
fn test_into_iter_drop_partial() {
    // exercise IntoIter::drop with remaining elements
    let lru = make_lru();
    let mut iter = lru.into_iter();
    let _ = iter.next();
    drop(iter);
}

#[test]
fn test_get_or_insert2_existed_and_inserted() {
    let mut lru = LRUCache::<String, String>::unbounded();

    let mut called = false;
    let v = lru.get_or_insert2("a", || {
        called = true;
        "Hello".to_owned()
    });
    assert!(called);
    v.push_str(", World!");

    // existed branch: closure not called
    let mut called = false;
    let v = lru.get_or_insert2("a", || {
        called = true;
        "nope".to_owned()
    });
    assert!(!called);
    assert_eq!(v, "Hello, World!");
}

#[test]
fn test_retain_removes_and_keeps() {
    let mut lru: LRUCache<i32, i32> = (0..5).map(|i| (i, i)).collect();
    lru.retain(|&k, _| k % 2 == 0);
    let keys: Vec<_> = lru.keys().copied().collect();
    assert_eq!(keys, vec![0, 2, 4]);
}
