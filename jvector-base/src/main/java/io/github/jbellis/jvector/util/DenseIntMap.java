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

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * A map (but not a Map) of int -&gt; T where the int keys are dense-ish and start at zero,
 * but the size of the map is not known in advance. This provides fast, concurrent
 * updates with zero lock contention on the common path.
 * <p>
 * "Dense-ish" means that space is allocated for all keys from 0 to the highest key, but
 * it is valid to have gaps in the keys. The value associated with "gap" keys is null.
 * <p>
 * <b>Storage layout.</b> The map uses a two-tier structure:
 * <ul>
 *   <li>A <em>base</em> {@link AtomicReferenceArray} sized from the constructor's
 *       {@code initialCapacity}. The base is <b>immortal</b> — allocated once, never
 *       resized, never copied — so reads and writes for keys below {@code initialCapacity}
 *       are a single volatile load + slot access (and, for writes, a CAS + an
 *       {@code AtomicInteger} increment). This matches the legacy implementation's
 *       read throughput exactly, and beats its write throughput because no
 *       {@link java.util.concurrent.locks.ReentrantReadWriteLock} traversal is required.</li>
 *   <li>A lazily-allocated <em>overflow</em> tier of fixed-size segments (1024 slots each)
 *       for keys at or beyond {@code initialCapacity}. Once a segment is installed it is
 *       never reallocated, so writes through the overflow are also lock-free on the steady
 *       state. Only spine growth + first-time segment install share a single
 *       {@code synchronized} block — the hot path never takes it.</li>
 * </ul>
 * Callers that know their upper bound on node count (e.g. a fixed dataset size) should
 * pass it as {@code initialCapacity} so all operations stay on the fast base path.
 */
public class DenseIntMap<T> implements IntMap<T> {
    private static final int OVERFLOW_SEGMENT_BITS = 10;
    private static final int OVERFLOW_SEGMENT_SIZE = 1 << OVERFLOW_SEGMENT_BITS;
    private static final int OVERFLOW_SEGMENT_MASK = OVERFLOW_SEGMENT_SIZE - 1;

    /** Immortal base array. Sized at construction and never reassigned. */
    private final AtomicReferenceArray<T> base;
    /** Cached {@code base.length()} so the hot path avoids the extra volatile read. */
    private final int baseLen;

    /** Lazily-installed segmented overflow for keys at or beyond {@link #baseLen}. */
    private volatile Overflow<T> overflow;
    private final Object overflowInitLock = new Object();

    private final AtomicInteger size = new AtomicInteger();

    public DenseIntMap(int initialCapacity) {
        if (initialCapacity <= 0) {
            throw new IllegalArgumentException("initialCapacity must be positive, got " + initialCapacity);
        }
        this.base = new AtomicReferenceArray<>(initialCapacity);
        this.baseLen = initialCapacity;
    }

    @Override
    public T get(int key) {
        if (key < 0) {
            return null;
        }
        if (key < baseLen) {
            return base.get(key);
        }
        Overflow<T> o = overflow;
        if (o == null) {
            return null;
        }
        return o.get(key - baseLen);
    }

    @Override
    public boolean compareAndPut(int key, T existing, T value) {
        if (value == null) {
            throw new IllegalArgumentException("compareAndPut() value cannot be null -- use remove() instead");
        }
        if (key < 0) {
            throw new IndexOutOfBoundsException("key must be non-negative, got " + key);
        }
        if (key < baseLen) {
            boolean success = base.compareAndSet(key, existing, value);
            if (success && existing == null) {
                size.incrementAndGet();
            }
            return success;
        }
        return overflowForWrite().compareAndPut(key - baseLen, existing, value, size);
    }

    @Override
    public int size() {
        return size.get();
    }

    @Override
    public T remove(int key) {
        if (key < 0) {
            return null;
        }
        if (key < baseLen) {
            T old = base.get(key);
            if (old == null) {
                return null;
            }
            if (base.compareAndSet(key, old, null)) {
                size.decrementAndGet();
                return old;
            }
            return null;
        }
        Overflow<T> o = overflow;
        if (o == null) {
            return null;
        }
        return o.remove(key - baseLen, size);
    }

    @Override
    public boolean containsKey(int key) {
        return get(key) != null;
    }

    @Override
    public void forEach(IntBiConsumer<T> consumer) {
        for (int i = 0; i < baseLen; i++) {
            T value = base.get(i);
            if (value != null) {
                consumer.consume(i, value);
            }
        }
        Overflow<T> o = overflow;
        if (o != null) {
            o.forEach(baseLen, consumer);
        }
    }

    private Overflow<T> overflowForWrite() {
        Overflow<T> o = overflow;
        if (o != null) {
            return o;
        }
        synchronized (overflowInitLock) {
            if (overflow == null) {
                overflow = new Overflow<>();
            }
            return overflow;
        }
    }

    /**
     * Segmented overflow tier for keys whose slot is not in {@link DenseIntMap#base}.
     * Keys are stored at relative offsets from {@code baseLen} (so the first overflow slot is
     * relKey 0). Segments are fixed-size and immortal once installed; the spine itself grows
     * under {@link #spineLock} but the hot path never touches it.
     */
    private static final class Overflow<T> {
        private volatile AtomicReferenceArray<AtomicReferenceArray<T>> spine;
        private final Object spineLock = new Object();

        Overflow() {
            this.spine = new AtomicReferenceArray<>(1);
        }

        T get(int relKey) {
            AtomicReferenceArray<AtomicReferenceArray<T>> spineSnap = spine;
            int segIdx = relKey >>> OVERFLOW_SEGMENT_BITS;
            if (segIdx >= spineSnap.length()) {
                return null;
            }
            AtomicReferenceArray<T> seg = spineSnap.get(segIdx);
            if (seg == null) {
                return null;
            }
            return seg.get(relKey & OVERFLOW_SEGMENT_MASK);
        }

        boolean compareAndPut(int relKey, T existing, T value, AtomicInteger size) {
            AtomicReferenceArray<T> seg = segmentFor(relKey);
            boolean success = seg.compareAndSet(relKey & OVERFLOW_SEGMENT_MASK, existing, value);
            if (success && existing == null) {
                size.incrementAndGet();
            }
            return success;
        }

        T remove(int relKey, AtomicInteger size) {
            AtomicReferenceArray<AtomicReferenceArray<T>> spineSnap = spine;
            int segIdx = relKey >>> OVERFLOW_SEGMENT_BITS;
            if (segIdx >= spineSnap.length()) {
                return null;
            }
            AtomicReferenceArray<T> seg = spineSnap.get(segIdx);
            if (seg == null) {
                return null;
            }
            int slot = relKey & OVERFLOW_SEGMENT_MASK;
            T old = seg.get(slot);
            if (old == null) {
                return null;
            }
            if (seg.compareAndSet(slot, old, null)) {
                size.decrementAndGet();
                return old;
            }
            return null;
        }

        void forEach(int baseOffset, IntBiConsumer<T> consumer) {
            AtomicReferenceArray<AtomicReferenceArray<T>> spineSnap = spine;
            for (int s = 0; s < spineSnap.length(); s++) {
                AtomicReferenceArray<T> seg = spineSnap.get(s);
                if (seg == null) {
                    continue;
                }
                int segBase = s << OVERFLOW_SEGMENT_BITS;
                for (int i = 0; i < OVERFLOW_SEGMENT_SIZE; i++) {
                    T value = seg.get(i);
                    if (value != null) {
                        consumer.consume(baseOffset + segBase + i, value);
                    }
                }
            }
        }

        private AtomicReferenceArray<T> segmentFor(int relKey) {
            int segIdx = relKey >>> OVERFLOW_SEGMENT_BITS;
            AtomicReferenceArray<AtomicReferenceArray<T>> spineSnap = spine;
            if (segIdx < spineSnap.length()) {
                AtomicReferenceArray<T> seg = spineSnap.get(segIdx);
                if (seg != null) {
                    return seg;
                }
            }
            synchronized (spineLock) {
                spineSnap = spine;
                if (segIdx >= spineSnap.length()) {
                    int newLen = Math.max(spineSnap.length() * 2, segIdx + 1);
                    AtomicReferenceArray<AtomicReferenceArray<T>> next = new AtomicReferenceArray<>(newLen);
                    for (int i = 0; i < spineSnap.length(); i++) {
                        next.set(i, spineSnap.get(i));
                    }
                    spineSnap = next;
                    spine = next;
                }
                AtomicReferenceArray<T> seg = spineSnap.get(segIdx);
                if (seg == null) {
                    seg = new AtomicReferenceArray<>(OVERFLOW_SEGMENT_SIZE);
                    spineSnap.set(segIdx, seg);
                }
                return seg;
            }
        }
    }
}
