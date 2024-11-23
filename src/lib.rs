#![doc = include_str!("../README.md")]
#![no_std]
#![deny(
    missing_docs,
    clippy::missing_safety_doc,
    clippy::undocumented_unsafe_blocks,
    clippy::must_use_candidate,
    clippy::perf,
    clippy::complexity,
    clippy::suspicious
)]

extern crate alloc;

use alloc::boxed::Box;
use core::borrow::Borrow;
use core::cell::UnsafeCell;

use intrusive_collections::intrusive_adapter;
use intrusive_collections::rbtree::Entry as RBTreeEntry;
use intrusive_collections::{
    KeyAdapter, LinkedList, LinkedListLink, RBTree, RBTreeLink, UnsafeRef,
};

struct Entry<K, V> {
    list_link: LinkedListLink,
    tree_link: RBTreeLink,
    key: K,
    value: UnsafeCell<V>,
}

impl<K, V> Entry<K, V> {
    #[inline(always)]
    fn new(key: K, value: V) -> UnsafeRef<Self> {
        UnsafeRef::from_box(Box::new(Self {
            list_link: LinkedListLink::new(),
            tree_link: RBTreeLink::new(),
            key,
            value: UnsafeCell::new(value),
        }))
    }

    #[inline(always)]
    fn value(&self) -> &V {
        // SAFETY: Read-only access to value is safe in conjunction with
        // the guarantees of other methods.
        unsafe { &*self.value.get() }
    }

    /// SAFETY: Only use with exclusive access to the Entry
    #[inline(always)]
    unsafe fn replace_value(&self, value: V) -> V {
        core::ptr::replace(self.value.get(), value)
    }
}

intrusive_adapter!(EntryListAdapter<K, V> = UnsafeRef<Entry<K, V>>: Entry<K, V> { list_link: LinkedListLink });
intrusive_adapter!(EntryTreeAdapter<K, V> = UnsafeRef<Entry<K, V>>: Entry<K, V> { tree_link: RBTreeLink });

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

impl<'a, K: 'a, V> KeyAdapter<'a> for EntryTreeAdapter<K, V> {
    type Key = Key<&'a K>; // Allows `Key<&K>: Borrow<Borrowed<Q>>`

    #[inline(always)]
    fn get_key(&self, value: &'a Entry<K, V>) -> Self::Key {
        // SAFETY: &K == Key<&K> == &Key<K> due to transparent repr
        unsafe { core::mem::transmute(&value.key) }
    }
}

/// LRU Cache implementation using intrusive collections.
///
/// This cache uses an [`intrusive_collections::LinkedList`] to maintain the LRU order,
/// and an [`intrusive_collections::RBTree`] to allow for efficient lookups by key,
/// while maintaining only one allocation per key-value pair. Unfortunately, this
/// is a linked structure, so cache locality is likely poor, but memory usage
/// and flexibility are improved.
///
/// The cache is unbounded by default, but can be limited to a maximum capacity.
///
/// # Example
/// ```rust
/// use intrusive_lru_cache::LRUCache;
///
/// let mut lru: LRUCache<&'static str, &'static str> = LRUCache::default();
///
/// lru.insert("a", "1");
/// lru.insert("b", "2");
/// lru.insert("c", "3");
///
/// let _ = lru.get("b"); // updates LRU order
///
/// assert_eq!(lru.pop(), Some(("a", "1")));
/// assert_eq!(lru.pop(), Some(("c", "3")));
/// assert_eq!(lru.pop(), Some(("b", "2")));
/// assert_eq!(lru.pop(), None);
/// ```
///
/// # Notes
///
/// - The cache is not thread-safe, and requires external synchronization.
/// - Cloning the cache will preserve the LRU order.
#[must_use]
pub struct LRUCache<K, V> {
    list: LinkedList<EntryListAdapter<K, V>>,
    tree: RBTree<EntryTreeAdapter<K, V>>,
    size: usize,
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
            size: 0,
            max_capacity,
        }
    }
}

impl<K, V> Default for LRUCache<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Clone for LRUCache<K, V>
where
    K: Clone + Ord + 'static,
    V: Clone,
{
    fn clone(&self) -> Self {
        let mut new = Self::new_with_max_capacity(self.max_capacity);

        // preserves the LRU ordering
        for (key, value) in self.iter_lru() {
            new.insert(key.clone(), value.clone());
        }

        new
    }
}

impl<K, V> LRUCache<K, V>
where
    K: Ord + 'static,
{
    /// Returns a reference to the value corresponding to the key,
    /// and bumps the key to the front of the LRU list.
    pub fn get<'a, Q>(&'a mut self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let entry = self.tree.find(Borrowed::new(key)).get()?;

        // SAFETY: Cursor created from a known valid pointer
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
    pub fn peek<'a, Q>(&'a self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
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
            // SAFETY: We treat the cursor as a mutable reference, and only use known valid pointers
            RBTreeEntry::Occupied(cursor) => unsafe {
                let entry = cursor.get().unwrap();

                // NOTE: Treat cursor/entry as if it were mutable for replace_value
                // since we can't ever actually acquire a mutable reference to the entry
                // as per the restrictions of `intrusive_collections`
                let old_value = entry.replace_value(value);

                // remove and reinsert at front to update LRU order
                let lru = self
                    .list
                    .cursor_mut_from_ptr(entry)
                    .remove()
                    .expect("tree and list are inconsistent");

                self.list.push_front(lru);

                Some(old_value)
            },
            RBTreeEntry::Vacant(cursor) => {
                let entry = Entry::new(key, value);

                cursor.insert(entry.clone());
                self.list.push_front(entry);

                self.size += 1;

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

        // SAFETY: Cursor created from a known valid pointer
        let _ = unsafe {
            self.list
                .cursor_mut_from_ptr(&*entry)
                .remove()
                .expect("tree and list are inconsistent")
        };

        self.size -= 1;

        // SAFETY: entry is removed from both the tree and list
        let Entry { value, .. } = unsafe { *UnsafeRef::into_box(entry) };

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
    #[inline(always)]
    pub fn set_max_capacity(&mut self, max_capacity: usize) {
        self.max_capacity = max_capacity;
    }

    /// Clears the cache, removing all key-value pairs.
    pub fn clear(&mut self) {
        self.tree.fast_clear();

        let mut front = self.list.front_mut();

        while let Some(entry) = front.remove() {
            // SAFETY: entry is removed from both the tree and list
            let _ = unsafe { UnsafeRef::into_box(entry) };
        }
    }

    /// Removes the oldest entries from the cache until the length is less than or equal to the maximum capacity.
    pub fn shrink(&mut self) {
        while self.size > self.max_capacity {
            let _ = self.pop();
        }
    }

    /// Removes the oldest entries from the cache until the length is less than or equal to the maximum capacity,
    /// and calls the provided closure with the removed key-value pairs.
    ///
    /// # Example
    /// ```rust
    /// # use intrusive_lru_cache::LRUCache;
    /// let mut lru: LRUCache<&'static str, &'static str> = LRUCache::default();
    ///
    /// lru.insert("a", "1");
    /// lru.insert("b", "2");
    /// lru.insert("c", "3");
    ///
    /// lru.set_max_capacity(1);
    ///
    /// let mut removed = Vec::new();
    ///
    /// lru.shrink_with(|key, value| {
    ///    removed.push((key, value));
    /// });
    ///
    /// assert_eq!(removed, vec![("a", "1"), ("b", "2")]);
    /// ```
    pub fn shrink_with<F>(&mut self, mut cb: F)
    where
        F: FnMut(K, V),
    {
        while self.size > self.max_capacity {
            let Some((key, value)) = self.pop() else {
                break;
            };

            cb(key, value);
        }
    }

    /// Returns the number of key-value pairs in the cache.
    #[inline(always)]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if the cache is empty.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        debug_assert_eq!(self.size == 0, self.list.is_empty());

        self.size == 0
    }

    /// Removes and returns the least recently used key-value pair.
    ///
    /// This is an `O(1)` operation.
    pub fn pop(&mut self) -> Option<(K, V)> {
        let entry = self.list.pop_back()?;

        // SAFETY: Cursor created from a known valid pointer
        let _ = unsafe {
            self.tree
                .cursor_mut_from_ptr(&*entry)
                .remove()
                .expect("tree and list are inconsistent")
        };

        self.size -= 1;

        // SAFETY: entry is removed from both the tree and list
        let Entry { key, value, .. } = unsafe { *UnsafeRef::into_box(entry) };

        Some((key, value.into_inner()))
    }

    /// Returns an iterator over the key-value pairs in the cache,
    /// in order of least recently used to most recently used.
    #[must_use]
    pub fn iter_lru(&self) -> impl DoubleEndedIterator<Item = (&K, &V)> {
        self.list.iter().map(|entry| (&entry.key, entry.value()))
    }

    /// Returns an iterator over the key-value pairs in the cache,
    /// in order of key `Ord` order.
    #[must_use]
    pub fn iter_ord(&self) -> impl DoubleEndedIterator<Item = (&K, &V)> {
        self.tree.iter().map(|entry| (&entry.key, entry.value()))
    }
}

impl<K, V> Drop for LRUCache<K, V> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<K, V> Extend<(K, V)> for LRUCache<K, V>
where
    K: Ord + 'static,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = (K, V)>,
    {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<K, V> FromIterator<(K, V)> for LRUCache<K, V>
where
    K: Ord + 'static,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (K, V)>,
    {
        let mut cache = Self::new();
        cache.extend(iter);
        cache
    }
}

/// An owning iterator over the key-value pairs in the cache,
/// in order of least recently used to most recently used.
pub struct IntoIter<K, V> {
    inner: intrusive_collections::linked_list::IntoIter<EntryListAdapter<K, V>>,
}

impl<K, V> IntoIterator for LRUCache<K, V>
where
    K: Ord + 'static,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(mut self) -> Self::IntoIter {
        self.tree.fast_clear();

        // swap out the list to avoid double drop
        let list = core::mem::replace(&mut self.list, LinkedList::new(EntryListAdapter::new()));

        IntoIter {
            inner: list.into_iter(),
        }
    }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        let entry = self.inner.next()?;

        // SAFETY: entry is removed from both the tree and list
        let Entry { key, value, .. } = unsafe { *UnsafeRef::into_box(entry) };

        Some((key, value.into_inner()))
    }
}

impl<K, V> DoubleEndedIterator for IntoIter<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let entry = self.inner.next_back()?;

        // SAFETY: entry is removed from both the tree and list
        let Entry { key, value, .. } = unsafe { *UnsafeRef::into_box(entry) };

        Some((key, value.into_inner()))
    }
}
