/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
package io.github.jbellis.jvector.bench;

import io.github.jbellis.jvector.util.ArrayUtil;
import io.github.jbellis.jvector.util.IntMap;
import io.github.jbellis.jvector.util.RamUsageEstimator;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Verbatim copy of the previous {@code DenseIntMap} implementation (volatile
 * {@link AtomicReferenceArray} + {@link ReentrantReadWriteLock} with Lucene-style
 * {@code ArrayUtil.oversize} growth). Kept in the benchmarks module so
 * {@link DenseIntMapConcurrentBenchmark} can measure the new segmented impl against
 * the old one under identical conditions, without needing a separate checkout.
 */
public class LegacyDenseIntMap<T> implements IntMap<T> {
    private final ReadWriteLock rwl = new ReentrantReadWriteLock();
    private volatile AtomicReferenceArray<T> objects;
    private final AtomicInteger size;

    public LegacyDenseIntMap(int initialCapacity) {
        objects = new AtomicReferenceArray<>(initialCapacity);
        size = new AtomicInteger();
    }

    @Override
    public boolean compareAndPut(int key, T existing, T value) {
        if (value == null) {
            throw new IllegalArgumentException("compareAndPut() value cannot be null -- use remove() instead");
        }

        ensureCapacity(key);
        rwl.readLock().lock();
        try {
            var success = objects.compareAndSet(key, existing, value);
            var isInsert = success && existing == null;
            if (isInsert) {
                size.incrementAndGet();
            }
            return success;
        } finally {
            rwl.readLock().unlock();
        }
    }

    @Override
    public int size() {
        return size.get();
    }

    @Override
    public T get(int key) {
        if (key >= objects.length()) {
            return null;
        }
        return objects.get(key);
    }

    private void ensureCapacity(int node) {
        if (node < objects.length()) {
            return;
        }

        rwl.writeLock().lock();
        try {
            var oldArray = objects;
            if (node >= oldArray.length()) {
                int newSize = ArrayUtil.oversize(node + 1, RamUsageEstimator.NUM_BYTES_OBJECT_REF);
                var newArray = new AtomicReferenceArray<T>(newSize);
                for (int i = 0; i < oldArray.length(); i++) {
                    newArray.set(i, oldArray.get(i));
                }
                objects = newArray;
            }
        } finally {
            rwl.writeLock().unlock();
        }
    }

    @Override
    public T remove(int key) {
        if (key >= objects.length()) {
            return null;
        }
        var old = objects.get(key);
        if (old == null) {
            return null;
        }

        rwl.readLock().lock();
        try {
            if (objects.compareAndSet(key, old, null)) {
                size.decrementAndGet();
                return old;
            } else {
                return null;
            }
        } finally {
            rwl.readLock().unlock();
        }
    }

    @Override
    public boolean containsKey(int key) {
        return get(key) != null;
    }

    @Override
    public void forEach(IntBiConsumer<T> consumer) {
        var ref = objects;
        for (int i = 0; i < ref.length(); i++) {
            var value = get(i);
            if (value != null) {
                consumer.consume(i, value);
            }
        }
    }
}
