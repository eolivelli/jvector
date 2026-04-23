/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.util;

import org.agrona.collections.Int2ObjectHashMap;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.StampedLock;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

/**
 * A concurrent {@link IntMap} backed by a striped array of Agrona {@link Int2ObjectHashMap}
 * shards.
 * <p>
 * Compared to a {@link java.util.concurrent.ConcurrentHashMap}{@code <Integer, T>}, this avoids
 * boxing the {@code int} key (no {@code Integer} per entry) and avoids the per-entry
 * {@code Node}/{@code TreeNode} overhead — both of which dominate the heap when storing tens of
 * millions of entries (see jvector#3).
 * <p>
 * <b>Concurrency.</b> Each shard is guarded by its own {@link StampedLock}. Reads
 * ({@link #get}, {@link #containsKey}) attempt a lock-free optimistic read first, falling back
 * to a read lock only if a writer interleaves; this matches the throughput of
 * {@link java.util.concurrent.ConcurrentHashMap}'s volatile-read fast path on uncontended
 * reads. Writes ({@link #compareAndPut}, {@link #remove}) take the shard write lock and are
 * serialised per shard. {@link #size()} is O(1) via a global {@link AtomicInteger}.
 * <p>
 * <b>Iteration.</b> {@link #forEach}, {@link #forEachKey} and {@link #keysStream} snapshot each
 * shard under its read lock into a primitive {@code int[]} (no boxing) plus an {@code Object[]}
 * of values, then release the lock before invoking the consumer. This (a) avoids deadlock if
 * the consumer re-enters the map and (b) preserves the weakly-consistent semantics that
 * callers had with the previous {@code ConcurrentHashMap}-backed implementation. There is no
 * global atomic snapshot — entries added or removed during iteration may or may not be visible.
 * <p>
 * <b>Footprint note.</b> This map is intended for HNSW upper layers (typically thousands to
 * millions of entries). The 32-shard structure has a fixed ~6 KB idle footprint, dwarfing the
 * cost of an empty map; do not use it for tiny maps.
 */
public class SparseIntMap<T> implements IntMap<T> {
    /** Number of shards; must be a power of two. */
    static final int SHARD_COUNT = 32;
    private static final int SHARD_MASK = SHARD_COUNT - 1;

    private final Int2ObjectHashMap<T>[] shards;
    private final StampedLock[] locks;
    private final AtomicInteger size = new AtomicInteger();

    @SuppressWarnings("unchecked")
    public SparseIntMap() {
        this.shards = (Int2ObjectHashMap<T>[]) new Int2ObjectHashMap[SHARD_COUNT];
        this.locks = new StampedLock[SHARD_COUNT];
        for (int i = 0; i < SHARD_COUNT; i++) {
            this.shards[i] = new Int2ObjectHashMap<>();
            this.locks[i] = new StampedLock();
        }
    }

    /**
     * Avalanche-mix the key before sharding. Agrona uses identity hashing internally, so a raw
     * monotonically-increasing key (typical in HNSW node IDs) would pile onto a single shard
     * without mixing.
     */
    static int shardIndex(int key) {
        int h = key;
        h ^= (h >>> 16);
        h *= 0x85EBCA6B;
        h ^= (h >>> 13);
        return h & SHARD_MASK;
    }

    @Override
    public boolean compareAndPut(int key, T existing, T value) {
        if (value == null) {
            throw new IllegalArgumentException("compareAndPut() value cannot be null -- use remove() instead");
        }

        int idx = shardIndex(key);
        StampedLock lock = locks[idx];
        long stamp = lock.writeLock();
        try {
            Int2ObjectHashMap<T> shard = shards[idx];
            T cur = shard.get(key);
            if (existing == null) {
                if (cur != null) {
                    return false;
                }
                shard.put(key, value);
                size.incrementAndGet();
                return true;
            }
            // Reference equality matches CHM.replace(k, expected, new) for our value types,
            // which do not override equals().
            if (cur != existing) {
                return false;
            }
            shard.put(key, value);
            return true;
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public int size() {
        return size.get();
    }

    @Override
    public T get(int key) {
        int idx = shardIndex(key);
        StampedLock lock = locks[idx];
        Int2ObjectHashMap<T> shard = shards[idx];

        // Optimistic read first — Agrona's open-addressing get may transiently observe a
        // resize-in-progress and return null or throw, so always validate and fall back to a
        // pessimistic read when the optimistic snapshot couldn't be confirmed.
        long stamp = lock.tryOptimisticRead();
        if (stamp != 0) {
            T value;
            try {
                value = shard.get(key);
            } catch (Throwable ignored) {
                value = null;
            }
            if (lock.validate(stamp)) {
                return value;
            }
        }
        stamp = lock.readLock();
        try {
            return shard.get(key);
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public T remove(int key) {
        int idx = shardIndex(key);
        StampedLock lock = locks[idx];
        long stamp = lock.writeLock();
        try {
            T old = shards[idx].remove(key);
            if (old != null) {
                size.decrementAndGet();
            }
            return old;
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public boolean containsKey(int key) {
        // Cheap and correct: get() already does the optimistic-read dance.
        return get(key) != null;
    }

    @Override
    public void forEach(IntBiConsumer<T> consumer) {
        for (int s = 0; s < SHARD_COUNT; s++) {
            int[] keys;
            Object[] values;
            StampedLock lock = locks[s];
            long stamp = lock.readLock();
            try {
                Int2ObjectHashMap<T> shard = shards[s];
                int n = shard.size();
                int[] kBuf = new int[n];
                Object[] vBuf = new Object[n];
                int[] pos = {0};
                shard.forEachInt((k, v) -> {
                    int p = pos[0];
                    if (p < kBuf.length) {
                        kBuf[p] = k;
                        vBuf[p] = v;
                        pos[0] = p + 1;
                    }
                });
                int filled = pos[0];
                if (filled == kBuf.length) {
                    keys = kBuf;
                    values = vBuf;
                } else {
                    keys = Arrays.copyOf(kBuf, filled);
                    values = Arrays.copyOf(vBuf, filled);
                }
            } finally {
                lock.unlockRead(stamp);
            }
            for (int i = 0; i < keys.length; i++) {
                @SuppressWarnings("unchecked")
                T v = (T) values[i];
                if (v != null) {
                    consumer.consume(keys[i], v);
                }
            }
        }
    }

    @Override
    public void forEachKey(IntConsumer consumer) {
        for (int s = 0; s < SHARD_COUNT; s++) {
            int[] keys = snapshotKeys(s);
            for (int k : keys) {
                consumer.accept(k);
            }
        }
    }

    @Override
    public IntStream keysStream() {
        int total = size.get();
        // Allocate slack to absorb concurrent inserts; growth handled per-shard below.
        int[] all = new int[Math.max(total + (total >> 3) + 16, 16)];
        int filled = 0;
        for (int s = 0; s < SHARD_COUNT; s++) {
            int[] keys = snapshotKeys(s);
            if (filled + keys.length > all.length) {
                int newLen = Math.max(all.length * 2, filled + keys.length);
                all = Arrays.copyOf(all, newLen);
            }
            System.arraycopy(keys, 0, all, filled, keys.length);
            filled += keys.length;
        }
        return Arrays.stream(all, 0, filled);
    }

    private int[] snapshotKeys(int shardIdx) {
        StampedLock lock = locks[shardIdx];
        long stamp = lock.readLock();
        try {
            Int2ObjectHashMap<T> shard = shards[shardIdx];
            int n = shard.size();
            int[] keys = new int[n];
            int[] pos = {0};
            shard.forEachInt((k, v) -> {
                int p = pos[0];
                if (p < keys.length) {
                    keys[p] = k;
                    pos[0] = p + 1;
                }
            });
            int filled = pos[0];
            return filled == keys.length ? keys : Arrays.copyOf(keys, filled);
        } finally {
            lock.unlockRead(stamp);
        }
    }
}
