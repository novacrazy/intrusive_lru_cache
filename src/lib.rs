#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

use alloc::rc::Rc;
use core::borrow::Borrow;
use core::cell::UnsafeCell;

use intrusive_collections::intrusive_adapter;
use intrusive_collections::rbtree::Entry as RBTreeEntry;
use intrusive_collections::{KeyAdapter, LinkedList, LinkedListLink, RBTree, RBTreeLink};

// Because KeyAdapter returns a reference, and `find` uses the returned type as `K`,
// I ran into issues where `&K: Borrow<Q>` was not satisfied. Therefore, we need
// to convince the compiler that some `Q` can be borrowed from `&K` by using a
// transparent wrapper type for both halves, and casting `&Q` to `&Borrowed<Q>`.

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
struct Key<K>(K);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
struct Borrowed<Q: ?Sized>(Q);

impl<'a, Q: ?Sized> Borrowed<Q> {
    #[inline(always)]
    const fn new(value: &'a Q) -> &'a Self {
        // SAFETY: &Q == &Borrowed<Q> due to transparent repr
        unsafe { core::mem::transmute(value) }
    }
}

// Magic that allows `&K: Borrow<Q>` to be satisfied
impl<K, Q: ?Sized> Borrow<Borrowed<Q>> for Key<&K>
where
    K: Borrow<Q>,
{
    #[inline(always)]
    fn borrow(&self) -> &Borrowed<Q> {
        Borrowed::new(self.0.borrow())
    }
}

struct Entry<K, V> {
    list_link: LinkedListLink,
    tree_link: RBTreeLink,
    key: K,
    value: UnsafeCell<V>,
}

impl<K, V> Entry<K, V> {
    #[inline(always)]
    fn new_rc(key: K, value: V) -> Rc<Self> {
        Rc::new(Self {
            list_link: LinkedListLink::new(),
            tree_link: RBTreeLink::new(),
            key,
            value: UnsafeCell::new(value),
        })
    }

    #[inline(always)]
    fn value(&self) -> &V {
        unsafe { &*self.value.get() }
    }

    /// SAFETY: Only use with exclusive access to the Entry
    #[inline(always)]
    unsafe fn replace_value(&self, value: V) -> V {
        core::ptr::replace(self.value.get(), value)
    }
}

intrusive_adapter!(EntryListAdapter<K, V> = Rc<Entry<K, V>>: Entry<K, V> { list_link: LinkedListLink });
intrusive_adapter!(EntryTreeAdapter<K, V> = Rc<Entry<K, V>>: Entry<K, V> { tree_link: RBTreeLink });

impl<'a, K: 'a, V> KeyAdapter<'a> for EntryTreeAdapter<K, V> {
    type Key = Key<&'a K>; // Allows `Key<&K>: Borrow<Borrowed<Q>>`

    #[inline(always)]
    fn get_key(&self, value: &'a Entry<K, V>) -> Self::Key {
        // SAFETY: &K == Key<&K> == &Key<K> due to transparent repr
        unsafe { core::mem::transmute(&value.key) }
    }
}

pub struct LRUCache<K, V> {
    list: LinkedList<EntryListAdapter<K, V>>,
    tree: RBTree<EntryTreeAdapter<K, V>>,
    len: usize,
    max_capacity: usize,
}

impl<K, V> LRUCache<K, V> {
    /// Creates a new unbounded LRU cache.
    ///
    /// This cache has no limit on the number of entries it can hold,
    /// so entries must be manually removed via [`pop`](Self::pop),
    /// or you can use [`set_max_capacity`](Self::set_max_capacity) to set a limit.
    pub fn new() -> Self {
        Self::new_with_max_capacity(usize::MAX)
    }

    /// Creates a new LRU cache with a maximum capacity, after which
    /// old entries will be evicted to make room for new ones.
    ///
    /// This does not preallocate any memory, only sets an upper limit.
    pub fn new_with_max_capacity(max_capacity: usize) -> Self {
        Self {
            list: LinkedList::new(EntryListAdapter::new()),
            tree: RBTree::new(EntryTreeAdapter::new()),
            len: 0,
            max_capacity,
        }
    }
}

impl<K, V> Default for LRUCache<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> LRUCache<K, V>
where
    K: Ord + 'static,
{
    /// Returns a reference to the value corresponding to the key,
    /// and bumps the key to the front of the LRU list.
    pub fn get<'a, 'b, Q>(&'a mut self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
        'a: 'b,
    {
        let entry = self.tree.find(Borrowed::new(key)).get()?;

        let cursor = unsafe {
            self.list
                .cursor_mut_from_ptr(entry)
                .remove()
                .expect("tree and list are inconsistent")
        };

        self.list.front_mut().insert_before(cursor);

        Some(entry.value())
    }

    /// Returns a reference to the value corresponding to the key,
    /// without updating the LRU list.
    pub fn peek<'a, 'b, Q>(&'a self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
        'a: 'b,
    {
        self.tree
            .find(Borrowed::new(key))
            .get()
            .map(|entry| entry.value())
    }

    /// Inserts a key-value pair into the cache, returning
    /// the old value if the key was already present.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.tree.entry(Borrowed::new(&key)) {
            RBTreeEntry::Occupied(cursor) => unsafe {
                let entry = cursor.get().unwrap();

                // NOTE: Treat cursor/entry as if it were mutable for replace_value
                // since we can't ever actually acquire a mutable reference to the entry
                // as per the restrictions of `intrusive_collections`
                let old_value = Some(entry.replace_value(value));

                let lru = self
                    .list
                    .cursor_mut_from_ptr(entry)
                    .remove()
                    .expect("tree and list are inconsistent");

                self.list.push_front(lru);

                old_value
            },
            RBTreeEntry::Vacant(cursor) => {
                let entry = Entry::new_rc(key, value);

                cursor.insert(entry.clone());
                self.list.push_front(entry);

                self.len += 1;

                self.shrink();

                None
            }
        }
    }

    /// Removes the value corresponding to the key from the cache,
    /// and returning it if it was present.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let entry = self.tree.find_mut(Borrowed::new(key)).remove()?;

        let _ = unsafe {
            self.list
                .cursor_mut_from_ptr(&*entry)
                .remove()
                .expect("tree and list are inconsistent")
        };

        self.len -= 1;

        let Ok(Entry { value, .. }) = Rc::try_unwrap(entry) else {
            unreachable!("tree and list are inconsistent")
        };

        Some(value.into_inner())
    }
}

impl<K, V> LRUCache<K, V> {
    /// Sets the maximum capacity of the cache.
    ///
    /// This does not remove any entries, but will cause the cache to evict
    /// entries when inserting new ones if the length exceeds the new capacity.
    ///
    /// Use [`shrink`](Self::shrink) to manually trigger removal of entries
    /// to meet the new capacity.
    pub fn set_max_capacity(&mut self, max_capacity: usize) {
        self.max_capacity = max_capacity;
    }

    /// Removes entries from the cache until the length is less than or equal to the maximum capacity.
    pub fn shrink(&mut self) {
        while self.len > self.max_capacity {
            let _ = self.pop();
        }
    }

    /// Returns the number of key-value pairs in the cache.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        debug_assert_eq!(self.len == 0, self.list.is_empty());

        self.len == 0
    }

    /// Removes and returns the least recently used key-value pair.
    pub fn pop(&mut self) -> Option<(K, V)> {
        let entry = self.list.pop_back()?;

        let _ = unsafe {
            self.tree
                .cursor_mut_from_ptr(&*entry)
                .remove()
                .expect("tree and list are inconsistent")
        };

        self.len -= 1;

        let Ok(Entry { key, value, .. }) = Rc::try_unwrap(entry) else {
            unreachable!("tree and list are inconsistent")
        };

        Some((key, value.into_inner()))
    }
}
