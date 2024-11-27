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

use alloc::borrow::ToOwned;
use alloc::boxed::Box;

use core::borrow::Borrow;
use core::cell::UnsafeCell;
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut, Index};
use core::ptr::NonNull;

use intrusive_collections::intrusive_adapter;
use intrusive_collections::rbtree::Entry as RBTreeEntry;
use intrusive_collections::{Bound, KeyAdapter, LinkedList, RBTree, UnsafeRef};

#[cfg(feature = "atomic")]
use intrusive_collections::{LinkedListAtomicLink as LinkedListLink, RBTreeAtomicLink as RBTreeLink};

#[cfg(not(feature = "atomic"))]
use intrusive_collections::{LinkedListLink, RBTreeLink};

// intrusive-collections does not provide a way to mutably access the value
// in a node, so we need to use `UnsafeCell` to allow for interior mutability.
#[repr(transparent)]
struct Value<V>(UnsafeCell<V>);

impl<V> Value<V> {
    #[inline(always)]
    const fn new(value: V) -> Self {
        Self(UnsafeCell::new(value))
    }

    #[inline(always)]
    const fn get(&self) -> &V {
        // SAFETY: Read-only access to value is safe in conjunction with
        // the guarantees of other methods.
        unsafe { &*self.0.get() }
    }

    // SAFETY: Only use with exclusive access to the Node
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    const unsafe fn get_mut(&self) -> &mut V {
        &mut *self.0.get()
    }

    /// SAFETY: Only use with exclusive access to the Node
    #[inline(always)]
    const unsafe fn replace(&self, value: V) -> V {
        core::ptr::replace(self.0.get(), value)
    }

    #[inline(always)]
    fn into_inner(self) -> V {
        self.0.into_inner()
    }
}

// SAFETY: Value is Send/Sync if V is Send/Sync,
// because the `Value<V>` is only accessed with exclusive access to the Node.
unsafe impl<V> Send for Value<V> where V: Send {}

// SAFETY: Value is Send/Sync if V is Send/Sync,
// because the `Value<V>` is only accessed with exclusive access to the Node.
unsafe impl<V> Sync for Value<V> where V: Sync {}

struct Node<K, V> {
    list_link: LinkedListLink,
    tree_link: RBTreeLink,
    key: K,
    value: Value<V>,
}

impl<K, V> Node<K, V> {
    #[inline(always)]
    fn new(key: K, value: V) -> UnsafeRef<Self> {
        UnsafeRef::from_box(Box::new(Self {
            list_link: LinkedListLink::new(),
            tree_link: RBTreeLink::new(),
            key,
            value: Value::new(value),
        }))
    }

    /// Assumes the node is not in any collections, and extracts the key/value
    /// when deallocating it.
    ///
    /// SAFETY: Only use after ensuring the node is not in any collections
    #[inline(always)]
    unsafe fn unwrap(this: UnsafeRef<Self>) -> (K, V) {
        let Node { key, value, .. } = *UnsafeRef::into_box(this);

        (key, value.into_inner())
    }
}

intrusive_adapter!(NodeListAdapter<K, V> = UnsafeRef<Node<K, V>>: Node<K, V> { list_link: LinkedListLink });
intrusive_adapter!(NodeTreeAdapter<K, V> = UnsafeRef<Node<K, V>>: Node<K, V> { tree_link: RBTreeLink });

// Because KeyAdapter returns a reference, and `find` uses the returned type as `K`,
// I ran into issues where `&K: Borrow<Q>` was not satisfied. Therefore, we need
// to convince the compiler that some `Q` can be borrowed from `&K` by using a
// transparent wrapper type for both halves, and casting `&Q` to `&Borrowed<Q>`.

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
struct Key<K: ?Sized>(K);

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

impl<'a, K: 'a, V> KeyAdapter<'a> for NodeTreeAdapter<K, V> {
    type Key = Key<&'a K>; // Allows `Key<&K>: Borrow<Borrowed<Q>>`

    #[inline(always)]
    fn get_key(&self, value: &'a Node<K, V>) -> Self::Key {
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
/// The `smart_*` methods allow for reading or updating the LRU order at the same time,
/// based on how the value is accessed. The `get` method always updates the LRU order,
/// and the `peek_*` methods allow for reading without updating the LRU order.
///
/// An overarching principle here is that the 'Used' in Lease-Recently-Used
/// is defined by the mutable access to the value. This allows for a more flexible
/// API, where the LRU order can be updated only when necessary, and not on every
/// read access.
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
/// - Cloning preserves LRU order.
/// - If the `atomic` crate feature is enabled,
///     the cache is thread-safe if `K` and `V` are `Send`/`Sync`.
#[must_use]
pub struct LRUCache<K, V> {
    list: LinkedList<NodeListAdapter<K, V>>,
    tree: RBTree<NodeTreeAdapter<K, V>>,
    size: usize,
    max_capacity: usize,
}

impl<K, V> fmt::Debug for LRUCache<K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter_peek_ord()).finish()
    }
}

impl<K, V> LRUCache<K, V> {
    /// Defines the size of the internal node structure in bytes.
    ///
    /// The memory footprint of the entire cache can then be calculated as:
    /// ```rust,ignore
    /// LRUCache::<K, V>::NODE_SIZE * cache.len()
    ///     + size_of::<LRUCache<K, V>>() // size of the cache itself
    /// ```
    /// or via [`memory_footprint`](Self::memory_footprint).
    ///
    /// This is a nice benefit of intrusive collections, as it allows for a single
    /// allocation per key-value pair.
    pub const NODE_SIZE: usize = size_of::<Node<K, V>>();

    /// Returns the total bytes consumed by the cache, including
    /// all allocations, internal structures, and the cache itself.
    #[must_use]
    #[inline]
    pub const fn memory_footprint(&self) -> usize {
        Self::NODE_SIZE * self.size + size_of::<Self>()
    }

    /// Creates a new unbounded LRU cache.
    ///
    /// This cache has no limit on the number of entries it can hold,
    /// so entries must be manually removed via [`pop`](Self::pop),
    /// or you can use [`set_max_capacity`](Self::set_max_capacity) to set a limit.
    #[inline]
    pub fn unbounded() -> Self {
        Self::new(usize::MAX)
    }

    /// Creates a new LRU cache with a maximum capacity, after which
    /// old entries will be evicted to make room for new ones.
    ///
    /// This does not preallocate any memory, only sets an upper limit.
    pub fn new(max_capacity: usize) -> Self {
        Self {
            list: LinkedList::new(NodeListAdapter::new()),
            tree: RBTree::new(NodeTreeAdapter::new()),
            size: 0,
            max_capacity,
        }
    }
}

impl<K, V> Default for LRUCache<K, V> {
    #[inline]
    fn default() -> Self {
        Self::unbounded()
    }
}

impl<K, V> Clone for LRUCache<K, V>
where
    K: Clone + Ord + 'static,
    V: Clone,
{
    fn clone(&self) -> Self {
        let mut new = Self::new(self.max_capacity);

        // preserves the LRU ordering by placing the oldest in first
        for (key, value) in self.iter_peek_lru().rev() {
            new.insert(key.clone(), value.clone());
        }

        new
    }
}

/// Bumps a node to the front of the list, only if it's not already there.
fn bump<K, V>(list: &mut LinkedList<NodeListAdapter<K, V>>, node: &Node<K, V>) {
    // SAFETY: The list is guaranteed to be non-empty  by virtue of `node` existing
    let front = unsafe { list.front().get().unwrap_unchecked() };

    // don't bother if it's already at the front
    if core::ptr::eq(node, front) {
        return;
    }

    // SAFETY: Cursor created from a known valid pointer
    let node = unsafe { list.cursor_mut_from_ptr(node).remove().unwrap_unchecked() };

    list.push_front(node);
}

impl<K, V> LRUCache<K, V>
where
    K: Ord + 'static,
{
    /// Returns a reference to the value corresponding to the key,
    /// without updating the LRU list.
    ///
    /// This is an `O(log n)` operation.
    pub fn peek<'a, Q>(&'a self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.tree
            .find(Borrowed::new(key))
            .get()
            .map(|node| node.value.get())
    }

    /// Returns a reference to the most recently used key-value pair,
    /// without updating the LRU order further.
    ///
    /// This is an `O(1)` operation.
    #[must_use]
    #[inline]
    pub fn peek_newest(&self) -> Option<(&K, &V)> {
        self.list.front().get().map(|node| (&node.key, node.value.get()))
    }

    /// Returns a reference to the least recently used key-value pair,
    /// without updating the LRU order further.
    ///
    /// This is an `O(1)` operation.
    #[must_use]
    #[inline]
    pub fn peek_oldest(&self) -> Option<(&K, &V)> {
        self.list.back().get().map(|node| (&node.key, node.value.get()))
    }

    /// Returns a reference to the most recently used key-value pair.
    ///
    /// Because this is already the most recently used, it's free to be accessed without
    /// any additional cost or additional modification of the LRU order.
    ///
    /// This is an `O(1)` operation.
    #[inline]
    pub fn get_newest(&mut self) -> Option<(&K, &mut V)> {
        self.list
            .front_mut()
            .into_ref()
            // SAFETY: We have `&mut self`
            .map(|node| unsafe { (&node.key, node.value.get_mut()) })
    }

    // SAFETY: The caller must guarantee that the cache is non-empty
    #[inline(always)]
    unsafe fn get_newest_unchecked(&mut self) -> (&K, &mut V) {
        let node = self.list.front_mut().into_ref().unwrap_unchecked();

        (&node.key, node.value.get_mut())
    }

    /// Returns a reference to the value corresponding to the key,
    /// and bumps the key to the front of the LRU list.
    ///
    /// This is an `O(log n)` operation.
    pub fn get<'a, Q>(&'a mut self, key: &Q) -> Option<&'a mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let node = self.tree.find(Borrowed::new(key)).get()?;

        bump(&mut self.list, node);

        // SAFETY: We have `&mut self`
        Some(unsafe { node.value.get_mut() })
    }

    /// If the key is present in the cache, it is bumped to the
    /// front of the LRU list as the most recently used. This
    /// will cause it to be the last to be evicted.
    ///
    /// Has no effect if the key is not present.
    ///
    /// This is an `O(log n)` operation.
    pub fn promote<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if let Some(node) = self.tree.find(Borrowed::new(key)).get() {
            bump(&mut self.list, node);
        }
    }

    /// If the key is present in the cache, it is demoted to the
    /// back of the LRU list as the least recently used. This
    /// will cause it to be evicted first if the cache is full.
    ///
    /// Has no effect if the key is not present.
    ///
    /// This is an `O(log n)` operation.
    pub fn demote<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if let Some(node) = self.tree.find(Borrowed::new(key)).get() {
            // same logic as bump, but for the back, and only used here

            // SAFETY: The list is guaranteed to be non-empty by virtue of `node` existing
            let back = unsafe { self.list.back().get().unwrap_unchecked() };

            // don't bother if it's already at the back
            if core::ptr::eq(node, back) {
                return;
            }

            // SAFETY: Cursor created from a known valid pointer
            let node = unsafe { self.list.cursor_mut_from_ptr(node).remove().unwrap_unchecked() };

            self.list.push_back(node);
        }
    }

    /// Returns a smart reference to the value corresponding to the key,
    /// allowing for reading and updating at the same time,
    /// only updating the LRU on mutable access.
    ///
    /// This does not immediately update the LRU order; only
    /// when the value is accessed via [`SmartEntry::get`] or
    /// [`SmartEntry::deref_mut`].
    ///
    /// Immutable access via [`SmartEntry::peek`] or [`SmartEntry::deref`]
    /// does not update the LRU order.
    ///
    /// This is an `O(log n)` operation.
    pub fn smart_get<'a, Q>(&'a mut self, key: &Q) -> Option<SmartEntry<'a, K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        Some(SmartEntry {
            node: self.tree.find(Borrowed::new(key)).get()?,
            list: NonNull::from(&mut self.list),
            _marker: PhantomData,
        })
    }

    /// Returns a smart reference to the least recently used key-value pair,
    /// allowing for reading and updating at the same time,
    /// only updating the LRU on mutable access.
    ///
    /// This does not immediately update the LRU order; only
    /// when the value is accessed via [`SmartEntry::get`] or
    /// [`SmartEntry::deref_mut`].
    ///
    /// This is an `O(1)` operation.
    pub fn smart_get_oldest(&mut self) -> Option<SmartEntry<'_, K, V>> {
        let list = NonNull::from(&mut self.list);

        Some(SmartEntry {
            node: self.list.back_mut().into_ref()?,
            list,
            _marker: PhantomData,
        })
    }

    /// Returns an iterator over the key-value pairs in the cache described by the range,
    /// _without_ updating the LRU order. The order of iteration is dependent on the order
    /// of the keys in the tree via `Ord`.
    ///
    /// The range follows `[min, max)`, where `min` is inclusive and `max` is exclusive.
    ///
    /// This is an `O(log n)` operation, though the iterator itself is `O(1)` to increment.
    pub fn peek_range<'a, MIN, MAX>(
        &'a self,
        min: &MIN,
        max: &MAX,
    ) -> impl DoubleEndedIterator<Item = (&'a K, &'a V)>
    where
        K: Borrow<MIN> + Borrow<MAX>,
        MIN: Ord + ?Sized,
        MAX: Ord + ?Sized,
    {
        self.tree
            .range(
                Bound::Included(Borrowed::new(min)),
                Bound::Excluded(Borrowed::new(max)),
            )
            .map(move |node| (&node.key, node.value.get()))
    }

    /// Returns an iterator over the key-value pairs in the cache described by the range,
    /// and **updates the LRU order** as they are yielded. The order of iteration is dependent
    /// on the order of the keys in the tree via `Ord`.
    ///
    /// The range follows `[min, max)`, where `min` is inclusive and `max` is exclusive.
    ///
    /// This is an `O(log n)` operation, though the iterator itself is `O(1)` to increment.
    pub fn range<'a, MIN, MAX>(
        &'a mut self,
        min: &MIN,
        max: &MAX,
    ) -> impl DoubleEndedIterator<Item = (&'a K, &'a mut V)>
    where
        K: Borrow<MIN> + Borrow<MAX>,
        MIN: Ord + ?Sized,
        MAX: Ord + ?Sized,
    {
        let LRUCache { tree, list, .. } = self;

        tree.range(
            Bound::Included(Borrowed::new(min)),
            Bound::Excluded(Borrowed::new(max)),
        )
        .map(move |node| {
            bump(list, node);

            // SAFETY: We have `&mut self`
            (&node.key, unsafe { node.value.get_mut() })
        })
    }

    /// Returns an iterator over the key-value pairs in the cache described by the range,
    /// which allows for reading and updating the LRU order at the same time, only
    /// bumping the LRU order when the value is accessed mutably via either [`SmartEntry::get`]
    /// or [`SmartEntry::deref_mut`].
    ///
    /// This is an `O(log n)` operation, though the iterator itself is `O(1)` to increment.
    pub fn smart_range<'a, MIN, MAX>(
        &'a mut self,
        min: &MIN,
        max: &MAX,
    ) -> impl DoubleEndedIterator<Item = SmartEntry<'a, K, V>>
    where
        K: Borrow<MIN> + Borrow<MAX>,
        MIN: Ord + ?Sized,
        MAX: Ord + ?Sized,
    {
        let LRUCache { tree, .. } = self;

        let list = NonNull::from(&mut self.list);

        tree.range(
            Bound::Included(Borrowed::new(min)),
            Bound::Excluded(Borrowed::new(max)),
        )
        .map(move |node| SmartEntry {
            node,
            list,
            _marker: PhantomData,
        })
    }

    /// Inserts a key-value pair into the cache, replacing
    /// the existing value if the key was already present, and then
    /// returning it. In both cases, the entry is moved to the front of the LRU list.
    ///
    /// This is an `O(log n)` operation.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.tree.entry(Borrowed::new(&key)) {
            // SAFETY: We treat the cursor as a mutable reference, and only use known valid pointers
            RBTreeEntry::Occupied(cursor) => unsafe {
                let node = cursor.get().unwrap_unchecked();

                // NOTE: Treat cursor/node as if it were mutable for value.replace
                // since we can't ever actually acquire a mutable reference to the node
                // as per the restrictions of `intrusive_collections`
                let old_value = node.value.replace(value);

                bump(&mut self.list, node);

                Some(old_value)
            },
            RBTreeEntry::Vacant(cursor) => {
                let node = Node::new(key, value);

                cursor.insert(node.clone());
                self.list.push_front(node);

                self.size += 1;
                self.shrink();

                None
            }
        }
    }

    /// Returns true if the cache contains the key.
    ///
    /// This does not update the LRU order.
    ///
    /// This is an `O(log n)` operation.
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        !self.tree.find(Borrowed::new(key)).is_null()
    }

    /// Removes the value corresponding to the key from the cache,
    /// and returning it if it was present. This has no effect on the order
    /// of other entries in the LRU list.
    ///
    /// This is an `O(log n)` operation.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let node = self.tree.find_mut(Borrowed::new(key)).remove()?;

        // SAFETY: Cursor created from a known valid pointer
        let _ = unsafe { self.list.cursor_mut_from_ptr(&*node).remove().unwrap_unchecked() };

        self.size -= 1;

        // SAFETY: node is removed from both the tree and list
        Some(unsafe { Node::unwrap(node).1 })
    }

    /// Attempts to get a mutable reference to the value corresponding to the key,
    /// and if it's not present, inserts a new key-value pair with the value returned
    /// by the provided closure. The entry is then moved to the front of the LRU list.
    ///
    /// This is similar to [`insert_or_get`](Self::insert_or_get), but for cases
    /// where the value is likely to be found in the cache already.
    ///
    /// See also [`get_or_insert2`](Self::get_or_insert2) for a version that allows for a borrowed key type.
    ///
    /// This is an `O(log n)` operation.
    pub fn get_or_insert<F>(&mut self, key: K, f: F) -> GetOrInsertResult<K, V>
    where
        F: FnOnce() -> V,
    {
        let key = match self.tree.entry(Borrowed::new(&key)) {
            // SAFETY: Cursor is a valid pointer here in both the tree and list
            RBTreeEntry::Occupied(cursor) => unsafe {
                bump(&mut self.list, cursor.get().unwrap_unchecked());

                Some(key)
            },
            RBTreeEntry::Vacant(cursor) => {
                let node = Node::new(key, f());

                cursor.insert(node.clone());
                self.list.push_front(node);

                self.size += 1;
                self.shrink();

                None
            }
        };

        // SAFETY: We have `&mut self` and the list is valid given the above logic
        // the element we want was _just_ repositioned to the front
        let v = unsafe { self.get_newest_unchecked().1 };

        match key {
            Some(key) => GetOrInsertResult::Existed(v, key),
            None => GetOrInsertResult::Inserted(v),
        }
    }

    /// Like [`get_or_insert`](Self::get_or_insert), but allows for a different borrowed key
    /// type, so long as it can be made into an owned key of type `K`.
    ///
    /// Because the key is borrowed initially, we can return `&mut V` instead of
    /// `GetOrInsertResult<K, V>`, as the key is not consumed.
    ///
    /// This is an `O(log n)` operation.
    ///
    /// # Example
    /// ```rust
    /// # use intrusive_lru_cache::LRUCache;
    /// let mut lru = LRUCache::<String, String>::unbounded();
    ///
    /// // note that the key is just an `&str`
    /// let v = lru.get_or_insert2("a", || "Hello".to_owned());
    /// v.push_str(", World!");
    ///
    /// assert_eq!(lru.pop().unwrap(), ("a".to_owned(), "Hello, World!".to_owned()));
    /// ```
    pub fn get_or_insert2<Q, F>(&mut self, key: &Q, f: F) -> &mut V
    where
        K: Borrow<Q>,
        Q: ToOwned<Owned = K> + Ord + ?Sized,
        F: FnOnce() -> V,
    {
        match self.tree.entry(Borrowed::new(key)) {
            // SAFETY: Cursor is a valid pointer here in both the tree and list
            RBTreeEntry::Occupied(cursor) => unsafe {
                bump(&mut self.list, cursor.get().unwrap_unchecked());
            },
            RBTreeEntry::Vacant(cursor) => {
                let node = Node::new(key.to_owned(), f());

                cursor.insert(node.clone());
                self.list.push_front(node);

                self.size += 1;
                self.shrink();
            }
        }

        // SAFETY: We have `&mut self` and the list is valid given the above logic
        // the element we want was _just_ repositioned to the front
        unsafe { self.get_newest_unchecked().1 }
    }

    /// Inserts a key-value pair into the cache only if it wasn't already present,
    /// otherwise update the LRU order for this element and return a reference to the value.
    ///
    /// The returned value contains a mutable reference to the value, and if the key already existed,
    /// it also contains the key and value that were passed in.
    ///
    /// This is similar to [`get_or_insert`](Self::get_or_insert), but for cases
    /// where insertion is expected.
    ///
    /// This is an `O(log n)` operation.
    pub fn insert_or_get(&mut self, key: K, value: V) -> InsertOrGetResult<'_, K, V> {
        let kv = match self.tree.entry(Borrowed::new(&key)) {
            // SAFETY: Cursor is a valid pointer here in both the tree and list
            RBTreeEntry::Occupied(cursor) => unsafe {
                let node = cursor.get().unwrap_unchecked();

                bump(&mut self.list, node);

                Some((key, value))
            },
            RBTreeEntry::Vacant(cursor) => {
                let node = Node::new(key, value);

                cursor.insert(node.clone());
                self.list.push_front(node);

                self.size += 1;

                self.shrink();

                None
            }
        };

        // SAFETY: We have `&mut self` and the list is valid given the above logic
        // the element we want was _just_ repositioned to the front
        let v = unsafe { self.get_newest_unchecked().1 };

        match kv {
            Some((key, value)) => InsertOrGetResult::Existed(v, key, value),
            None => InsertOrGetResult::Inserted(v),
        }
    }
}

/// The result of [`LRUCache::get_or_insert`](LRUCache::get_or_insert).
///
/// If inserted, it returns a reference to the newly inserted value.
/// If the key already existed, it returns a reference to the existing value, and the provided key.
#[derive(Debug, PartialEq, Eq)]
pub enum GetOrInsertResult<'a, K, V> {
    /// Element was inserted, key and value were consumed.
    Inserted(&'a mut V),

    /// Element already existed at the given key, so a reference
    /// to the existing value is returned, along with the given key.
    Existed(&'a mut V, K),
}

/// The result of [`LRUCache::insert_or_get`](LRUCache::insert_or_get).
///
/// If inserted, it returns a reference to the newly inserted value.
/// If the key already existed, it returns a reference to the existing value, the key and the value.
#[derive(Debug, PartialEq, Eq)]
pub enum InsertOrGetResult<'a, K, V> {
    /// Element was inserted, key and value were consumed.
    Inserted(&'a mut V),

    /// Element already existed at the given key, so a reference
    /// to the existing value is returned, along with the given key and value.
    Existed(&'a mut V, K, V),
}

impl<'a, K, V> GetOrInsertResult<'a, K, V> {
    /// Consumes the result and returns a reference to the value.
    ///
    /// This will drop the key if it existed.
    #[inline(always)]
    pub fn into_inner(self) -> &'a mut V {
        match self {
            Self::Inserted(value) => value,
            Self::Existed(value, _) => value,
        }
    }
}

impl<'a, K, V> InsertOrGetResult<'a, K, V> {
    /// Consumes the result and returns a reference to the value.
    ///
    /// This will drop the key and value if they existed.
    #[inline(always)]
    pub fn into_inner(self) -> &'a mut V {
        match self {
            Self::Inserted(value) => value,
            Self::Existed(value, _, _) => value,
        }
    }
}

impl<K, V> Deref for GetOrInsertResult<'_, K, V> {
    type Target = V;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Inserted(value) => value,
            Self::Existed(value, _) => value,
        }
    }
}

impl<K, V> Deref for InsertOrGetResult<'_, K, V> {
    type Target = V;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Inserted(value) => value,
            Self::Existed(value, _, _) => value,
        }
    }
}

impl<K, V> DerefMut for GetOrInsertResult<'_, K, V> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Inserted(value) => value,
            Self::Existed(value, _) => value,
        }
    }
}

impl<K, V> DerefMut for InsertOrGetResult<'_, K, V> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Inserted(value) => value,
            Self::Existed(value, _, _) => value,
        }
    }
}

impl<K, V> LRUCache<K, V> {
    /// Removes and returns the least recently used key-value pair.
    ///
    /// This is an `O(1)` operation.
    pub fn pop(&mut self) -> Option<(K, V)> {
        let node = self.list.pop_back()?;

        // SAFETY: Cursor created from a known valid pointer
        let _ = unsafe { self.tree.cursor_mut_from_ptr(&*node).remove().unwrap_unchecked() };

        self.size -= 1;

        // SAFETY: node is removed from both the tree and list
        Some(unsafe { Node::unwrap(node) })
    }

    // TODO: Add a pop_while function that takes a filter callback and returns an iterator

    /// Removes and returns the highest `Ord` key-value pair.
    ///
    /// This is an `O(1)` operation.
    pub fn pop_highest(&mut self) -> Option<(K, V)> {
        let node = self.tree.back_mut().remove()?;

        // SAFETY: Cursor created from a known valid pointer
        let _ = unsafe { self.list.cursor_mut_from_ptr(&*node).remove().unwrap_unchecked() };

        self.size -= 1;

        // SAFETY: node is removed from both the tree and list
        Some(unsafe { Node::unwrap(node) })
    }

    /// Removes and returns the lowest `Ord` key-value pair.
    ///
    /// This is an `O(1)` operation.
    pub fn pop_lowest(&mut self) -> Option<(K, V)> {
        let node = self.tree.front_mut().remove()?;

        // SAFETY: Cursor created from a known valid pointer
        let _ = unsafe { self.list.cursor_mut_from_ptr(&*node).remove().unwrap_unchecked() };

        self.size -= 1;

        // SAFETY: node is removed from both the tree and list
        Some(unsafe { Node::unwrap(node) })
    }
}

impl<K, V> LRUCache<K, V> {
    /// Sets the maximum capacity of the cache.
    ///
    /// **This does not remove any entries**, but will cause the cache to evict
    /// entries when inserting new ones if the length exceeds the new capacity.
    ///
    /// Use [`shrink`](Self::shrink) to manually trigger removal of entries
    /// to meet the new capacity.
    #[inline(always)]
    pub fn set_max_capacity(&mut self, max_capacity: usize) {
        self.max_capacity = max_capacity;
    }

    /// Returns the maximum capacity of the cache, which is the
    /// point at which the cache will start evicting older entries.
    #[inline(always)]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.max_capacity
    }

    /// Sets the maximum capacity of the cache, and removes the oldest entries
    /// until the length is less than or equal to the new capacity.
    ///
    /// If the new capacity is greater than the current length, no entries are removed.
    #[inline]
    pub fn resize(&mut self, max_capacity: usize) {
        self.set_max_capacity(max_capacity);
        self.shrink();
    }

    /// Clears the cache, removing all key-value pairs.
    pub fn clear(&mut self) {
        self.tree.fast_clear();

        let mut front = self.list.front_mut();

        while let Some(node) = front.remove() {
            // SAFETY: node is removed from both the tree and list
            let _ = unsafe { UnsafeRef::into_box(node) };
        }
    }

    /// Removes the oldest entries from the cache until the length is less than or equal to the maximum capacity.
    pub fn shrink(&mut self) {
        while self.size > self.max_capacity {
            let _ = self.pop();
        }
    }

    /// Removes up to `amount` of the oldest entries from the cache.
    pub fn shrink_by(&mut self, amount: usize) {
        for _ in 0..amount {
            if self.pop().is_none() {
                break;
            }
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

    /// Removes up to `amount` of the oldest entries from the cache,
    /// and calls the provided closure with the removed key-value pairs.
    pub fn shrink_by_with<F>(&mut self, amount: usize, mut cb: F)
    where
        F: FnMut(K, V),
    {
        for _ in 0..amount {
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

    /// Returns `true` if the cache is full.
    ///
    /// Attempting to insert a new key-value pair will cause the cache to evict at least
    /// one entry to make room for the new one.
    #[inline(always)]
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.size >= self.max_capacity
    }

    /// Returns an iterator over the keys in the cache,
    /// in order of key `Ord` order.
    ///
    /// NOTE: This does _not_ update the LRU order.
    #[must_use]
    pub fn keys(&self) -> impl DoubleEndedIterator<Item = &K> {
        self.tree.iter().map(|node| &node.key)
    }

    /// Returns an iterator over immutable key-value pairs in the cache,
    /// in order of most recently used to least recently used.
    ///
    /// NOTE: This does _not_ update the LRU order.
    #[must_use]
    pub fn iter_peek_lru(&self) -> impl DoubleEndedIterator<Item = (&K, &V)> {
        self.list.iter().map(|node| (&node.key, node.value.get()))
    }

    /// Returns an iterator over immutable key-value pairs in the cache,
    /// in order of key `Ord` order.
    ///
    /// NOTE: This does _not_ update the LRU order.
    #[must_use]
    pub fn iter_peek_ord(&self) -> impl DoubleEndedIterator<Item = (&K, &V)> {
        self.tree.iter().map(|node| (&node.key, node.value.get()))
    }

    /// Returns an iterator yielding key-value pairs in the cache as [`SmartEntry`],
    /// in the order determined by the `Ord` implementation of the keys. This allows for
    /// reading and updating the LRU order at the same time, only updating the LRU order
    /// when the value is accessed mutably.
    ///
    /// Entries yielded do not immediately update the LRU order; only when the value is accessed
    /// via [`SmartEntry::get`] or [`SmartEntry::deref_mut`].
    pub fn smart_iter(&mut self) -> impl DoubleEndedIterator<Item = SmartEntry<'_, K, V>> {
        let list = NonNull::from(&mut self.list);

        self.tree.iter().map(move |node| SmartEntry {
            node,
            list,
            _marker: PhantomData,
        })
    }

    /// Iterates over all key-value pairs in the cache, and calls the provided closure
    /// to determine if the key-value pair should be retained. If the closure returns `false`,
    /// the key-value pair is removed from the cache.
    ///
    /// LRU Order is unchanged.
    ///
    /// This is an `O(n)` operation.
    pub fn retain<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        let mut cursor = self.tree.front_mut();

        while let Some(node) = cursor.get() {
            if predicate(&node.key, node.value.get()) {
                cursor.move_next();
                continue;
            }

            // SAFETY: Cursor created from a known valid node
            unsafe { self.list.cursor_mut_from_ptr(node).remove().unwrap_unchecked() };

            // SAFETY: We were just using the node, but now we own it
            let node = unsafe { cursor.remove().unwrap_unchecked() };

            self.size -= 1;

            // SAFETY: node is removed from both the tree and list
            let _ = unsafe { UnsafeRef::into_box(node) };
        }
    }
}

/// An entry in the cache that can be used for for reading or writing,
/// only updating the LRU order when the value is accessed mutably.
///
/// The `Deref` and `DerefMut` implementations allow for easy access to the value,
/// without or with updating the LRU order, respectively. Accessing the value mutably
/// via `DerefMut` will update the LRU order.
///
/// See [`SmartEntry::peek`] and [`SmartEntry::get`] for more information.
#[must_use]
pub struct SmartEntry<'a, K, V> {
    node: &'a Node<K, V>,

    /// Since `Iterator` can't return a reference to self, we need to store the list
    /// as a pointer to be able to update the LRU order. For all intents and purposes,
    /// this pointer is equivalent to `&mut LinkedList<EntryListAdapter<K, V>>`.
    list: NonNull<LinkedList<NodeListAdapter<K, V>>>,
    _marker: core::marker::PhantomData<&'a mut LinkedList<NodeListAdapter<K, V>>>,
}

impl<K, V> Deref for SmartEntry<'_, K, V> {
    type Target = V;

    /// Dereferences the value, without updating the LRU order.
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.peek_value()
    }
}

impl<K, V> DerefMut for SmartEntry<'_, K, V> {
    /// Mutably dereferences the value, and updates the LRU order.
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_value()
    }
}

impl<K, V> SmartEntry<'_, K, V> {
    /// Access the key only, without updating the LRU order.
    #[inline(always)]
    #[must_use]
    pub fn key(&self) -> &K {
        &self.node.key
    }

    /// Access the key-value pair immutably, without updating the LRU order.
    #[inline(always)]
    #[must_use]
    pub fn peek(&self) -> (&K, &V) {
        (&self.node.key, self.node.value.get())
    }

    /// Access the value immutably, without updating the LRU order.
    ///
    /// This is the same as [`SmartEntry::deref`].
    #[inline(always)]
    #[must_use]
    pub fn peek_value(&self) -> &V {
        self.node.value.get()
    }

    /// Access the key-value pair, and update the LRU order.
    ///
    /// The LRU order is updated every time this method is called,
    /// as it is assumed that the caller is actively using the value.
    #[must_use]
    pub fn get(&mut self) -> (&K, &mut V) {
        // SAFETY: We tied the lifetime of the pointer to 'a, the same as the LRUCache,
        // so it will always be valid here. Furthermore, because it's a raw pointer,
        // SmartEntry is not Send/Sync, so as long as the mutability happens right
        // here and now, it's safe, same as an `&mut LinkedList`.
        bump(unsafe { self.list.as_mut() }, self.node);

        // SAFETY: We have exclusive access to the Node
        unsafe { (&self.node.key, self.node.value.get_mut()) }
    }

    /// Access the value mutably, and update the LRU order.
    ///
    /// This LRU order is updated every time this method is called,
    /// as it is assumed that the caller is actively using the value.
    ///
    /// The `DerefMut` implementation invokes this method to access the value,
    /// updating the LRU order in the process.
    #[inline(always)]
    pub fn get_value(&mut self) -> &mut V {
        self.get().1
    }
}

impl<K, V> Drop for LRUCache<K, V> {
    #[inline]
    fn drop(&mut self) {
        self.clear();
    }
}

impl<K, V, Q> Index<&Q> for LRUCache<K, V>
where
    K: Borrow<Q> + Ord + 'static,
    Q: Ord + ?Sized,
{
    type Output = V;

    /// Returns a reference to the value corresponding to the key,
    /// without updating the LRU order.
    ///
    /// # Panics
    ///
    /// Panics if the key is not present in the cache.
    #[inline]
    fn index(&self, index: &Q) -> &Self::Output {
        self.peek(index).expect("no entry found for key")
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
        let mut cache = Self::unbounded();
        cache.extend(iter);
        cache
    }
}

/// An owning iterator over the key-value pairs in the cache,
/// in order of most recently used to least recently used.
pub struct IntoIter<K, V> {
    list: LinkedList<NodeListAdapter<K, V>>,
}

impl<K, V> IntoIterator for LRUCache<K, V>
where
    K: Ord + 'static,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    #[inline]
    fn into_iter(mut self) -> Self::IntoIter {
        self.tree.fast_clear();

        IntoIter {
            list: self.list.take(),
        }
    }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: node is removed from the list
        self.list.pop_front().map(|node| unsafe { Node::unwrap(node) })
    }
}

impl<K, V> DoubleEndedIterator for IntoIter<K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        // SAFETY: node is removed from the list
        self.list.pop_back().map(|node| unsafe { Node::unwrap(node) })
    }
}

impl<K, V> Drop for IntoIter<K, V> {
    #[inline]
    fn drop(&mut self) {
        let mut front = self.list.front_mut();

        while let Some(node) = front.remove() {
            // SAFETY: node is removed from both the tree and list
            let _ = unsafe { UnsafeRef::into_box(node) };
        }
    }
}
