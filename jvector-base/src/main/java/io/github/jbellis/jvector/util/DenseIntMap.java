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
 * but the size of the map is not known in advance.  This provides fast, concurrent
 * updates and minimizes contention when the map is resized.
 * <p>
 * "Dense-ish" means that space is allocated for all keys from 0 to the highest key, but
 * it is valid to have gaps in the keys.  The value associated with "gap" keys is null.
 * <p>
 * Storage layout: a two-level structure consisting of a <em>spine</em> of fixed-size
 * <em>segments</em>. Once a segment is installed it is never reallocated, so all slot
 * reads and CAS updates are lock-free. Only the spine itself ever grows, and spine
 * grow + segment install share a single {@code synchronized} block — paths that find
 * an already-installed segment never touch that lock. This eliminates the
 * read/write-lock contention that the previous single-array design exhibited under
 * heavy concurrent {@code compareAndPut}, where any in-flight resize would park all
 * concurrent updaters under a non-fair AQS.
 */
public class DenseIntMap<T> implements IntMap<T> {
    private static final int SEGMENT_BITS = 10;
    private static final int SEGMENT_SIZE = 1 << SEGMENT_BITS;
    private static final int SEGMENT_MASK = SEGMENT_SIZE - 1;

    /** Spine of segments. Volatile so spine grows are visible without locking. */
    private volatile AtomicReferenceArray<AtomicReferenceArray<T>> spine;

    /** Serialises spine grow and segment install — never acquired on the steady-state hot path. */
    private final Object spineLock = new Object();

    private final AtomicInteger size = new AtomicInteger();

    public DenseIntMap(int initialCapacity) {
        if (initialCapacity <= 0) {
            throw new IllegalArgumentException("initialCapacity must be positive, got " + initialCapacity);
        }
        int nSegs = segmentIndex(initialCapacity - 1) + 1;
        AtomicReferenceArray<AtomicReferenceArray<T>> initial = new AtomicReferenceArray<>(nSegs);
        // Eagerly allocate all segments up to the requested capacity so callers that pass an
        // accurate size hint get zero allocation traffic during their hot insert phase.
        for (int i = 0; i < nSegs; i++) {
            initial.set(i, new AtomicReferenceArray<>(SEGMENT_SIZE));
        }
        this.spine = initial;
    }

    @Override
    public boolean compareAndPut(int key, T existing, T value) {
        if (value == null) {
            throw new IllegalArgumentException("compareAndPut() value cannot be null -- use remove() instead");
        }
        if (key < 0) {
            throw new IndexOutOfBoundsException("key must be non-negative, got " + key);
        }

        AtomicReferenceArray<T> seg = segmentFor(key, true);
        boolean success = seg.compareAndSet(key & SEGMENT_MASK, existing, value);
        if (success && existing == null) {
            size.incrementAndGet();
        }
        return success;
    }

    @Override
    public int size() {
        return size.get();
    }

    @Override
    public T get(int key) {
        if (key < 0) {
            return null;
        }
        AtomicReferenceArray<AtomicReferenceArray<T>> spineSnap = spine;
        int segIdx = key >>> SEGMENT_BITS;
        if (segIdx >= spineSnap.length()) {
            return null;
        }
        AtomicReferenceArray<T> seg = spineSnap.get(segIdx);
        if (seg == null) {
            return null;
        }
        return seg.get(key & SEGMENT_MASK);
    }

    @Override
    public T remove(int key) {
        if (key < 0) {
            return null;
        }
        AtomicReferenceArray<AtomicReferenceArray<T>> spineSnap = spine;
        int segIdx = key >>> SEGMENT_BITS;
        if (segIdx >= spineSnap.length()) {
            return null;
        }
        AtomicReferenceArray<T> seg = spineSnap.get(segIdx);
        if (seg == null) {
            return null;
        }
        int slot = key & SEGMENT_MASK;
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

    @Override
    public boolean containsKey(int key) {
        return get(key) != null;
    }

    @Override
    public void forEach(IntBiConsumer<T> consumer) {
        AtomicReferenceArray<AtomicReferenceArray<T>> spineSnap = spine;
        for (int s = 0; s < spineSnap.length(); s++) {
            AtomicReferenceArray<T> seg = spineSnap.get(s);
            if (seg == null) {
                continue;
            }
            int base = s << SEGMENT_BITS;
            for (int i = 0; i < SEGMENT_SIZE; i++) {
                T value = seg.get(i);
                if (value != null) {
                    consumer.consume(base + i, value);
                }
            }
        }
    }

    private static int segmentIndex(int key) {
        return key >>> SEGMENT_BITS;
    }

    /**
     * Resolve (and optionally install) the segment that owns {@code key}. The fast path is a
     * single volatile read of {@code spine} followed by an array slot read; only when the
     * segment is missing or the spine is too short do we acquire {@code spineLock}.
     */
    private AtomicReferenceArray<T> segmentFor(int key, boolean create) {
        int segIdx = key >>> SEGMENT_BITS;
        AtomicReferenceArray<AtomicReferenceArray<T>> spineSnap = spine;
        if (segIdx < spineSnap.length()) {
            AtomicReferenceArray<T> seg = spineSnap.get(segIdx);
            if (seg != null) {
                return seg;
            }
        }
        if (!create) {
            return null;
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
                seg = new AtomicReferenceArray<>(SEGMENT_SIZE);
                spineSnap.set(segIdx, seg);
            }
            return seg;
        }
    }
}
